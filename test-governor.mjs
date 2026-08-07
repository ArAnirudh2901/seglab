#!/usr/bin/env bun
/**
 * Unit tests for js/memory-governor.js — the 2.2 GB ceiling policy.
 *
 * The point of these is that the ceiling holds in the cases that actually
 * break it: a 45 MP file open while three heavy encoders want to be resident.
 *
 * Usage: bun test-governor.mjs
 */
import * as g from './js/memory-governor.js'

let pass = 0
let fail = 0
const ok = (label, cond, detail = '') => {
    if (cond) { pass += 1; return }
    fail += 1
    console.log(`✗ ${label}${detail ? ` — ${detail}` : ''}`)
}
const eq = (label, got, want) => ok(label, JSON.stringify(got) === JSON.stringify(want), `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`)

/* Baseline accounting. */
g.reset()
eq('empty budget uses only the reserve', g.budget().usedMB, 300)
ok('empty budget has spendable room', g.budget().spendableMB === 2200 - 300 - 250)

/* A 24 MP photo. */
g.reset()
g.setImageFootprint(6000, 4000)
ok('24MP image costs ~183 MB', g.budget().imageMB === 183, `${g.budget().imageMB} MB`)

/* A 45 MP photo — the case that breaks a naive policy. */
g.reset()
g.setImageFootprint(8000, 5600)
const big = g.budget()
ok('45MP image costs ~342 MB', big.imageMB > 300 && big.imageMB < 400, `${big.imageMB} MB`)

/* Three heavy encoders must never all be resident with a big file open. */
{
    g.reset()
    g.setImageFootprint(8000, 5600)
    g.register('draft', () => {})
    g.register('flagship', () => {})
    ok('text is NOT affordable beside the flagship at 45MP', !g.canAfford('text'),
        `spendable ${g.budget().spendableMB} MB vs text ${g.RESIDENT_COST.text} MB`)

    let disposed = null
    g.reset()
    g.setImageFootprint(8000, 5600)
    g.register('draft', () => { disposed = 'draft' })
    g.register('flagship', () => { disposed = 'flagship' })
    const made = await g.makeRoomFor('text', { protect: ['draft'] })
    ok('makeRoomFor evicts the flagship, not the draft', made && disposed === 'flagship', `made=${made} disposed=${disposed}`)
    ok('draft survives the eviction', g.isResident('draft'))
    ok('flagship is gone', !g.isResident('flagship'))
}

/* The ceiling actually holds after the swap. */
{
    g.reset()
    g.setImageFootprint(8000, 5600)
    g.register('draft', () => {})
    await g.makeRoomFor('text', { protect: ['draft'] })
    g.register('text', () => {})
    const b = g.budget()
    ok('total stays under the ceiling with a 45MP file + draft + text',
        b.usedMB <= b.ceilingMB, `${b.usedMB} MB vs ${b.ceilingMB} MB`)
    ok('safety margin is still intact', b.spendableMB >= 0, `spendable ${b.spendableMB} MB`)
}

/* Protecting everything means the load must be refused, not forced. */
{
    g.reset()
    g.configure({ ceilingMB: 1200 })
    g.setImageFootprint(8000, 5600)
    g.register('draft', () => {})
    g.register('flagship', () => {})
    const made = await g.makeRoomFor('text', { protect: ['draft', 'flagship'] })
    ok('refuses to load when nothing may be evicted', made === false)
}

/* A small machine shrinks the ceiling; an unknown one does not inflate it. */
{
    g.reset()
    g.adoptDeviceCeiling({ deviceMemory: 4 })
    ok('4 GB device shrinks the ceiling', g.budget().ceilingMB < 2200, `${g.budget().ceilingMB} MB`)

    g.reset()
    g.adoptDeviceCeiling({ deviceMemory: 64 })
    ok('64 GB device does not inflate past the configured ceiling', g.budget().ceilingMB === 2200, `${g.budget().ceilingMB} MB`)

    g.reset()
    g.adoptDeviceCeiling({})
    ok('unknown deviceMemory keeps the default', g.budget().ceilingMB === 2200)
}

/* Embedding cache sizing adapts to what is left. */
{
    g.reset()
    g.setImageFootprint(6000, 4000)
    g.register('text', () => {})
    const roomy = g.embeddingCacheMax(8.4, 6)
    ok('draft embeddings still cache several images', roomy >= 2 && roomy <= 6, `${roomy}`)

    g.reset()
    g.configure({ ceilingMB: 900 })
    g.setImageFootprint(6000, 4000)
    g.register('text', () => {})
    ok('a tight budget still allows at least one cached embedding', g.embeddingCacheMax(33, 2) >= 1)
}

/* Evicting something absent is a no-op, not a throw. */
g.reset()
await g.evict('nothing-here')
ok('evicting an absent resident is safe', true)

console.log(`\n${fail ? '✗ FAILED' : '✓ PASS'} — ${pass}/${pass + fail} assertions\n`)
process.exit(fail ? 1 : 0)
