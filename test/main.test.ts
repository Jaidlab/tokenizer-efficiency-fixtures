import {expect, it} from 'bun:test'

import {getMainModuleDefault} from 'zeug'

const tokenizerEfficiencyFixtures = await getMainModuleDefault<typeof import('tokenizer-efficiency-fixtures')>('tokenizer-efficiency-fixtures')

it('should run', () => {
  expect(tokenizerEfficiencyFixtures).toBe(1) // TODO Test actual functionality
})
