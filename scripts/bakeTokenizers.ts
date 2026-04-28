/* eslint-disable typescript/no-restricted-imports -- Keep this script aligned with the existing build scripts. */
import type {TiktokenBPE} from 'js-tiktoken/lite'

import {Buffer} from 'node:buffer'
import {mkdir, readFile, rm, writeFile} from 'node:fs/promises'
import * as path from 'node:path/posix'

import {parse as parseYaml} from 'yaml'

type VocabularyInfo = {
  huggingfaceRepo?: string
  import?: string
  name: string
}

type VocabularyMap = Record<string, VocabularyInfo>

type TokenizerJson = {
  added_tokens?: Array<{content: string
    id: number
    special?: boolean}>
  model?: {
    merges?: Array<[string, string] | string>
    type?: string
    vocab?: Record<string, number>
  }
  pre_tokenizer?: unknown
}

type TokenizerConfigJson = {
  added_tokens_decoder?: Record<string, {
    content: string
    special?: boolean
  }>
}

type BakedTokenizer = {
  details: Array<string>
  tokenizer: TiktokenBPE
}

type BakeResult = {
  details: Array<string>
  file?: string
  id: string
  name: string
  skipped?: boolean
}

const huggingFaceCacheRoot = 'temp/huggingface'
const outputRoot = 'out/tokenizer'
const tiktokenModelPatStr = String.raw`[\p{Script=Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
function range(start: number, end: number) {
  return Array.from({length: end - start + 1}, (_, index) => start + index)
}
const byteLevelDecoder = (() => {
  const bytes = [
    ...range(33, 126),
    ...range(161, 172),
    ...range(174, 255),
  ]
  const chars = [...bytes]
  let extra = 0
  for (let byte = 0; byte < 256; byte++) {
    if (bytes.includes(byte)) {
      continue
    }
    bytes.push(byte)
    chars.push(256 + extra)
    extra++
  }
  return new Map(chars.map((char, index) => [String.fromCodePoint(char), bytes[index]]))
})()
function formatNumber(value: number | undefined) {
  if (value === undefined) {
    return '–'
  }
  return String(value).replaceAll(/\B(?=(\d{3})+(?!\d))/g, ' ')
}
function bytesKey(bytes: Uint8Array) {
  return [...bytes].join(',')
}
function bytesToBase64(bytes: Uint8Array) {
  return Buffer.from(bytes).toString('base64')
}
function chunk<T>(items: Array<T>, size: number) {
  return Array.from({length: Math.ceil(items.length / size)}, (_, index) => items.slice(index * size, (index + 1) * size))
}
function buildTiktokenRanks(tokens: Array<Uint8Array>) {
  return chunk(tokens.map(bytesToBase64), 1024)
    .map((tokenChunk, index) => `! ${index * 1024} ${tokenChunk.join(' ')}`)
    .join('\n')
}
function countTiktokenRanks(ranks: string) {
  return ranks
    .split('\n')
    .filter(Boolean)
    .reduce((sum, line) => sum + Math.max(0, line.split(/\s+/).length - 2), 0)
}
function getSplitRegexes(preTokenizer: unknown): Array<string> {
  if (!preTokenizer || typeof preTokenizer !== 'object') {
    return []
  }
  const object = preTokenizer as Record<string, unknown>
  if (object.type === 'Sequence' && Array.isArray(object.pretokenizers)) {
    return object.pretokenizers.flatMap(getSplitRegexes)
  }
  if (object.type !== 'Split') {
    return []
  }
  const pattern = object.pattern
  if (!pattern || typeof pattern !== 'object') {
    return []
  }
  const regex = (pattern as Record<string, unknown>).Regex
  return typeof regex === 'string' ? [regex] : []
}
function hasByteLevelPreTokenizer(preTokenizer: unknown): boolean {
  if (!preTokenizer || typeof preTokenizer !== 'object') {
    return false
  }
  const object = preTokenizer as Record<string, unknown>
  if (object.type === 'ByteLevel') {
    return true
  }
  if (object.type === 'Sequence' && Array.isArray(object.pretokenizers)) {
    return object.pretokenizers.some(hasByteLevelPreTokenizer)
  }
  return false
}
function getPatStr(preTokenizer: unknown) {
  const regexes = getSplitRegexes(preTokenizer)
  if (regexes.length === 0) {
    return String.raw`[\s\S]+`
  }
  if (regexes.length === 1) {
    return regexes[0]
  }
  return regexes.map(regex => `(?:${regex})`).join('|')
}
function getSpecialTokens(tokenizerJson: TokenizerJson) {
  return Object.fromEntries((tokenizerJson.added_tokens ?? []).filter(token => token.special).map(token => [token.content, token.id]))
}
function getSpecialTokensFromConfig(tokenizerConfig: TokenizerConfigJson) {
  return Object.fromEntries(Object.entries(tokenizerConfig.added_tokens_decoder ?? {})
    .filter(([, token]) => token.special)
    .map(([id, token]) => [token.content, Number(id)]))
}
function getSortedVocabularyTokens(vocab: Record<string, number>) {
  return Object.entries(vocab).toSorted((a, b) => a[1] - b[1])
}
function byteLevelTokenToBytes(token: string) {
  const bytes: Array<number> = []
  for (const char of token) {
    const byte = byteLevelDecoder.get(char)
    if (byte === undefined) {
      return
    }
    bytes.push(byte)
  }
  return Uint8Array.from(bytes)
}
function buildByteLevelTiktokenBpe(tokenizerJson: TokenizerJson): TiktokenBPE {
  const vocab = tokenizerJson.model?.vocab
  const merges = tokenizerJson.model?.merges
  if (!vocab || !merges) {
    throw new Error('Hugging Face tokenizer.json does not contain model.vocab and model.merges.')
  }
  const assignedKeys = new Set<string>
  const orderedTokens: Array<Uint8Array> = []
  const assign = (token: string) => {
    const bytes = byteLevelTokenToBytes(token)
    if (!bytes) {
      return
    }
    const key = bytesKey(bytes)
    if (assignedKeys.has(key)) {
      return
    }
    assignedKeys.add(key)
    orderedTokens.push(bytes)
  }
  for (const [token] of getSortedVocabularyTokens(vocab)) {
    const bytes = byteLevelTokenToBytes(token)
    if (bytes?.length === 1) {
      assign(token)
    }
  }
  for (const merge of merges) {
    const parts = Array.isArray(merge) ? merge : merge.split(' ')
    assign(parts.join(''))
  }
  for (const [token] of getSortedVocabularyTokens(vocab)) {
    assign(token)
  }
  return {
    bpe_ranks: buildTiktokenRanks(orderedTokens),
    pat_str: getPatStr(tokenizerJson.pre_tokenizer),
    special_tokens: getSpecialTokens(tokenizerJson),
  }
}
function buildTiktokenModelBpe(modelText: string, tokenizerConfig: TokenizerConfigJson): TiktokenBPE {
  const ranks = modelText
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      const [token, rank] = line.split(/\s+/)
      return `! ${rank} ${token}`
    })
    .join('\n')
  return {
    bpe_ranks: ranks,
    pat_str: tiktokenModelPatStr,
    special_tokens: getSpecialTokensFromConfig(tokenizerConfig),
  }
}
function renderGeneratedTokenizer(tokenizer: TiktokenBPE, vocabulary: VocabularyInfo) {
  if (tokenizer.bpe_ranks.includes('`') || tokenizer.bpe_ranks.includes('${')) {
    throw new Error('bpe_ranks contains a sequence that is unsafe for a template literal.')
  }
  return `import type {TiktokenBPE} from 'js-tiktoken/lite'

// Generated by scripts/bakeTokenizers.ts from ${JSON.stringify(vocabulary.name)}.
export default {
  pat_str: ${JSON.stringify(tokenizer.pat_str)},
  special_tokens: ${JSON.stringify(tokenizer.special_tokens, null, 2).replaceAll('\n', '\n  ')},
  bpe_ranks: \`${tokenizer.bpe_ranks}\`,
} satisfies TiktokenBPE
`
}
async function writeTextFile(file: string, content: string) {
  await mkdir(path.dirname(file), {recursive: true})
  await writeFile(file, content)
}
function isFileNotFoundError(error: unknown) {
  return error instanceof Error && 'code' in error && error.code === 'ENOENT'
}
function getHuggingFaceCacheFile(repo: string, file: string) {
  return `${huggingFaceCacheRoot}/${repo.replaceAll('/', '--')}/${file}`
}
async function readCachedHuggingFaceFile(repo: string, file: string) {
  const cacheFile = getHuggingFaceCacheFile(repo, file)
  try {
    return await readFile(cacheFile, 'utf8')
  } catch (error) {
    if (!isFileNotFoundError(error)) {
      throw error
    }
  }
}
async function getCachedHuggingFaceFile(repo: string, file: string) {
  const cachedText = await readCachedHuggingFaceFile(repo, file)
  if (cachedText !== undefined) {
    return cachedText
  }
  const response = await fetch(`https://huggingface.co/${repo}/resolve/main/${file}`)
  if (!response.ok) {
    throw new Error(`Could not fetch ${repo}/${file}: HTTP ${response.status}.`)
  }
  const text = await response.text()
  await writeTextFile(getHuggingFaceCacheFile(repo, file), text)
  return text
}
async function loadImportedTokenizer(vocabulary: VocabularyInfo): Promise<BakedTokenizer> {
  if (!vocabulary.import) {
    throw new Error('Vocabulary does not define an import.')
  }
  const tokenizer = (await import(vocabulary.import) as {default: TiktokenBPE}).default
  return {
    details: [`Copied native js-tiktoken vocabulary export from ${vocabulary.import}.`],
    tokenizer,
  }
}
async function loadHuggingFaceTokenizer(vocabulary: VocabularyInfo): Promise<BakedTokenizer | undefined> {
  if (!vocabulary.huggingfaceRepo) {
    throw new Error('Vocabulary does not define a Hugging Face repository.')
  }
  const tokenizerConfig = JSON.parse(await getCachedHuggingFaceFile(vocabulary.huggingfaceRepo, 'tokenizer_config.json')) as TokenizerConfigJson
  const cachedModelText = await readCachedHuggingFaceFile(vocabulary.huggingfaceRepo, 'tiktoken.model')
  let tokenizerJsonText = await readCachedHuggingFaceFile(vocabulary.huggingfaceRepo, 'tokenizer.json')
  if (!tokenizerJsonText && !cachedModelText) {
    try {
      tokenizerJsonText = await getCachedHuggingFaceFile(vocabulary.huggingfaceRepo, 'tokenizer.json')
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes('HTTP 404')) {
        throw error
      }
    }
  }
  if (!tokenizerJsonText) {
    const modelText = cachedModelText ?? await getCachedHuggingFaceFile(vocabulary.huggingfaceRepo, 'tiktoken.model')
    const rankCount = modelText.split('\n').filter(Boolean).length
    return {
      details: [`Converted Hugging Face tiktoken.model from ${vocabulary.huggingfaceRepo}.`, `${formatNumber(rankCount)} ranks.`],
      tokenizer: buildTiktokenModelBpe(modelText, tokenizerConfig),
    }
  }
  const tokenizerJson = JSON.parse(tokenizerJsonText) as TokenizerJson
  if (tokenizerJson.model?.type !== 'BPE') {
    throw new Error(`Unsupported Hugging Face tokenizer model type: ${tokenizerJson.model?.type ?? 'unknown'}.`)
  }
  if (!hasByteLevelPreTokenizer(tokenizerJson.pre_tokenizer)) {
    return
  }
  const tokenizer = buildByteLevelTiktokenBpe(tokenizerJson)
  return {
    details: [`Converted Hugging Face ByteLevel BPE tokenizer from ${vocabulary.huggingfaceRepo}.`, `${formatNumber(countTiktokenRanks(tokenizer.bpe_ranks))} ranks.`],
    tokenizer,
  }
}
async function loadTokenizer(vocabulary: VocabularyInfo) {
  if (vocabulary.import) {
    return loadImportedTokenizer(vocabulary)
  }
  if (vocabulary.huggingfaceRepo) {
    return loadHuggingFaceTokenizer(vocabulary)
  }
  throw new Error('Vocabulary must define either “import” or “huggingfaceRepo”.')
}
async function bakeTokenizer(id: string, vocabulary: VocabularyInfo): Promise<BakeResult> {
  const bakedTokenizer = await loadTokenizer(vocabulary)
  if (!bakedTokenizer) {
    return {
      details: ['Skipped because the tokenizer is not backed by a ByteLevel BPE or tiktoken.model file, so a js-tiktoken rank export would not be faithful.'],
      id,
      name: vocabulary.name,
      skipped: true,
    }
  }
  const file = `${outputRoot}/${id}.ts`
  await writeTextFile(file, renderGeneratedTokenizer(bakedTokenizer.tokenizer, vocabulary))
  return {
    details: bakedTokenizer.details,
    file,
    id,
    name: vocabulary.name,
  }
}
const vocabularyMap = parseYaml(await readFile('src/vocabularies.yml', 'utf8')) as VocabularyMap
await rm(outputRoot, {
  force: true,
  recursive: true,
})
await mkdir(outputRoot, {recursive: true})
const results: Array<BakeResult> = []
for (const [id, vocabulary] of Object.entries(vocabularyMap)) {
  results.push(await bakeTokenizer(id, vocabulary))
}
const bakedResults = results.filter(result => !result.skipped)
const skippedResults = results.filter(result => result.skipped)
for (const result of results) {
  const icon = result.skipped ? '◦' : '✓'
  const target = result.file ? ` → ${result.file}` : ''
  console.log(`${icon} ${result.id} (${result.name})${target}`)
  for (const detail of result.details) {
    console.log(`  ${detail}`)
  }
}
console.log(`Baked ${formatNumber(bakedResults.length)} tokenizers in ${outputRoot}.`)
if (skippedResults.length) {
  console.log(`Skipped ${formatNumber(skippedResults.length)} tokenizer${skippedResults.length === 1 ? '' : 's'}.`)
}
