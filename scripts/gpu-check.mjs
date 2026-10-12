// Runs the GPU solver's checks (src/gpu/gpuChecks.ts) in headless Chrome with
// hardware WebGPU. Needs the dev server: `npm run dev`, then `npm run test:gpu`.
// Usage: node scripts/gpu-check.mjs [chrome path] [dev server url]
import { spawn } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const chromePath = process.argv[2] ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const url = process.argv[3] ?? 'http://localhost:5173/'
const profile = mkdtempSync(join(tmpdir(), 'gpu-check-'))
const port = 9300 + Math.floor(Math.random() * 500)
const chrome = spawn(chromePath, ['--headless=new', `--user-data-dir=${profile}`, `--remote-debugging-port=${port}`, 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let exitCode = 1
try {
  let targets
  for (let i = 0; i < 50 && !targets; i++) {
    try {
      targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json()
    } catch {
      await sleep(200)
    }
  }
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl)
  await new Promise((r) => (ws.onopen = r))
  let id = 0
  const pending = new Map()
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data)
    if (d.method === 'Runtime.consoleAPICalled') console.log(d.params.args.map((a) => a.value ?? a.description).join(' '))
    if (pending.has(d.id)) {
      pending.get(d.id)(d)
      pending.delete(d.id)
    }
  }
  const send = (method, params = {}) => new Promise((r) => {
    const i = ++id
    pending.set(i, r)
    ws.send(JSON.stringify({ id: i, method, params }))
  })
  await send('Runtime.enable')
  await send('Page.navigate', { url })
  await sleep(1500)
  const res = await send('Runtime.evaluate', {
    expression: `import('/src/gpu/gpuChecks.ts').then((m) => m.runGpuChecks()).then((r) => r.every((c) => c.ok))`,
    awaitPromise: true,
    returnByValue: true,
  })
  if (res.result.exceptionDetails) console.log(res.result.exceptionDetails.exception?.description)
  exitCode = res.result.result?.value === true ? 0 : 1
  ws.close()
} finally {
  chrome.kill()
  await sleep(300)
  rmSync(profile, { recursive: true, force: true })
}
process.exit(exitCode)
