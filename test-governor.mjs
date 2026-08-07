#!/usr/bin/env bun
/**
 * Unit tests for js/memory-governor.js — the 1 GB ceiling policy.
 *
 * The point of these is that the ceiling holds in the case that actually
 * breaks it: a 45 MP file decoded in the same tab as the model. Running one
 * model instead of a stack is what makes 1 GB reachable, but the photo alone
 * is a third of the budget, so the accounting has to include it.
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
eq('default ceiling is 1 GB', g.budget().ceilingMB, 1000)
eq('empty budget uses only the reserve', g.budget().usedMB, 250)
ok('empty budget has spendable room', g.budget().spendableMB === 1000 - 250 - 120)

/* The model fits on an empty budget. */
g.reset()
ok('the model is affordable with no photo open', g.canAfford('yoloe'),
    `spendable ${g.budget().spendableMB} vs ${g.RESIDENT_COST.yoloe}`)

/* A 24 MP photo. */
g.reset()
g.setImageFootprint(6000, 4000)
eq('24MP image costs ~183 MB', g.budget().imageMB, 183)
ok('model still fits beside a 24MP photo', g.canAfford('yoloe'),
    `spendable ${g.budget().spendableMB}`)

/* A 45 MP photo — the case that decides whether 1 GB is real. */
{
    g.reset()
    g.setImageFootprint(8000, 5600)
    const b = g.budget()
    ok('45MP image costs ~342 MB', b.imageMB > 300 && b.imageMB < 400, `${b.imageMB} MB`)
    ok('model still fits beside a 45MP photo', g.canAfford('yoloe'),
        `spendable ${b.spendableMB} vs ${g.RESIDENT_COST.yoloe}`)

    g.register('yoloe', () => {})
    const after = g.budget()
    ok('total stays under the ceiling with a 45MP file + model',
        after.usedMB <= after.ceilingMB, `${after.usedMB} MB vs ${after.ceilingMB} MB`)
    ok('safety margin survives the worst case', after.spendableMB >= 0, `spendable ${after.spendableMB} MB`)
}

/* An absurd image must be refused rather than forced. */
{
    g.reset()
    g.setImageFootprint(16000, 12000) // ~1.5 GB of RGBA
    ok('a 192MP image makes the model unaffordable', !g.canAfford('yoloe'),
        `spendable ${g.budget().spendableMB}`)
    const made = await g.makeRoomFor('yoloe')
    ok('makeRoomFor refuses when nothing can be freed', made === false)
}

/* Registering, evicting, re-registering. */
{
    g.reset()
    let disposed = false
    g.register('yoloe', () => { disposed = true })
    ok('resident is tracked', g.isResident('yoloe'))
    await g.evict('yoloe')
    ok('dispose hook ran', disposed)
    ok('resident is gone', !g.isResident('yoloe'))
    ok('budget released', g.budget().residentMB === 0)
}

/* An already-resident model needs no room made. */
{
    g.reset()
    g.setImageFootprint(8000, 5600)
    g.register('yoloe', () => {})
    ok('makeRoomFor is a no-op when already resident', await g.makeRoomFor('yoloe') === true)
    ok('and it was not evicted', g.isResident('yoloe'))
}

/* A small machine shrinks the ceiling; an unknown one does not inflate it. */
{
    g.reset()
    g.adoptDeviceCeiling({ deviceMemory: 2 })
    ok('2 GB device shrinks the ceiling', g.budget().ceilingMB < 1000, `${g.budget().ceilingMB} MB`)
    ok('but never below the 500 MB floor', g.budget().ceilingMB >= 500, `${g.budget().ceilingMB} MB`)

    g.reset()
    g.adoptDeviceCeiling({ deviceMemory: 64 })
    eq('64 GB device does not inflate past the configured ceiling', g.budget().ceilingMB, 1000)

    g.reset()
    g.adoptDeviceCeiling({})
    eq('unknown deviceMemory keeps the default', g.budget().ceilingMB, 1000)
}

/* Cache sizing adapts to what is left. */
{
    g.reset()
    g.setImageFootprint(6000, 4000)
    g.register('yoloe', () => {})
    const n = g.embeddingCacheMax(30, 4)
    ok('cache sizing stays within the hard max', n >= 1 && n <= 4, `${n}`)

    g.reset()
    g.configure({ ceilingMB: 600 })
    g.setImageFootprint(6000, 4000)
    g.register('yoloe', () => {})
    ok('a tight budget still allows at least one cached entry', g.embeddingCacheMax(200, 4) >= 1)
}

/* Evicting something absent is a no-op, not a throw. */
g.reset()
await g.evict('nothing-here')
ok('evicting an absent resident is safe', true)

console.log(`\n${fail ? '✗ FAILED' : '✓ PASS'} — ${pass}/${pass + fail} assertions\n`)
process.exit(fail ? 1 : 0)
