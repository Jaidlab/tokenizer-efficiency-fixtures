import {Buffer} from 'node:buffer'
import {mkdir, readFile, rm, writeFile} from 'node:fs/promises'
import * as path from 'node:path/posix'

import {globby} from 'globby'
import {Tiktoken} from 'js-tiktoken/lite'
import {parse as parseYaml, stringify as stringifyYaml} from 'yaml'
import {renderHandlebars} from 'zeug'

type Candidate = unknown

type FixtureInfo = {
  candidates: Array<Candidate>
  text?: string
}

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
  normalizer?: unknown
  pre_tokenizer?: unknown
}

type TokenizerConfigJson = {
  added_tokens_decoder?: Record<string, {
    content: string
    special?: boolean
  }>
}

type TokenEncoder = {
  countTokens: (text: string) => number
  details: Array<string>
  id: string
  name: string
  source: string
}

type VocabularyReport = {
  details: Array<string>
  error?: string
  id: string
  name: string
  source: string
}

type SentenceReport = {
  candidate: string
  counts: Record<string, number | undefined>
  minimum?: number
  text: string
}

type FixtureReport = {
  folderName: string
  sentenceReports: Array<SentenceReport>
  title: string
  totals: Record<string, number | undefined>
}

const huggingFaceCacheRoot = 'temp/huggingface'
const textEncoder = new TextEncoder
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
function range(start: number, end: number) {
  return Array.from({length: end - start + 1}, (_, index) => start + index)
}
function html(value: unknown) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}
function attribute(value: unknown) {
  return html(value)
}
function formatNumber(value: number | undefined) {
  if (value === undefined) {
    return '–'
  }
  return String(value).replaceAll(/\B(?=(\d{3})+(?!\d))/g, ' ')
}
function candidateToString(candidate: Candidate) {
  if (typeof candidate === 'string') {
    return candidate
  }
  if (candidate === null || typeof candidate !== 'object') {
    return String(candidate)
  }
  return JSON.stringify(candidate)
}
function getCandidateContext(candidate: Candidate) {
  if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
    return {
      ...candidate,
      candidate,
    }
  }
  return {candidate}
}
function resolveCandidateText(info: FixtureInfo, candidate: Candidate) {
  if (!info.text) {
    return candidateToString(candidate)
  }
  return renderHandlebars(info.text, getCandidateContext(candidate))
}
function getFixtureTitle(folderName: string) {
  return folderName
}
function getSpecialTokens(tokenizerJson: TokenizerJson) {
  return Object.fromEntries((tokenizerJson.added_tokens ?? []).filter(token => token.special).map(token => [token.content, token.id]))
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
function bytesKey(bytes: Uint8Array) {
  return [...bytes].join(',')
}
function bytesToBase64(bytes: Uint8Array) {
  return Buffer.from(bytes).toString('base64')
}
function buildTiktokenRanks(tokens: Array<Uint8Array>) {
  return tokens.map((bytes, rank) => `! ${rank} ${bytesToBase64(bytes)}`).join('\n')
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
function applyNormalizer(text: string, normalizer: unknown): string {
  if (!normalizer || typeof normalizer !== 'object') {
    return text
  }
  const object = normalizer as Record<string, unknown>
  if (object.type === 'NFC') {
    return text.normalize('NFC')
  }
  if (object.type === 'Sequence' && Array.isArray(object.normalizers)) {
    return object.normalizers.reduce((current, child) => applyNormalizer(current, child), text)
  }
  if (object.type === 'Replace' && object.pattern && typeof object.pattern === 'object' && typeof object.content === 'string') {
    const pattern = object.pattern as Record<string, unknown>
    if (typeof pattern.String === 'string') {
      return text.replaceAll(pattern.String, object.content)
    }
    if (typeof pattern.Regex === 'string') {
      return text.replaceAll(new RegExp(pattern.Regex, 'gu'), object.content)
    }
  }
  return text
}
function getSortedVocabularyTokens(vocab: Record<string, number>) {
  return Object.entries(vocab).toSorted((a, b) => a[1] - b[1])
}
function buildByteLevelTiktokenEncoder(id: string, vocabulary: VocabularyInfo, tokenizerJson: TokenizerJson): TokenEncoder {
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
  const tiktoken = new Tiktoken({
    bpe_ranks: buildTiktokenRanks(orderedTokens),
    pat_str: getPatStr(tokenizerJson.pre_tokenizer),
    special_tokens: getSpecialTokens(tokenizerJson),
  })
  return {
    countTokens: text => tiktoken.encode(applyNormalizer(text, tokenizerJson.normalizer), [], []).length,
    details: ['Hugging Face ByteLevel BPE converted to js-tiktoken ranks.', `${formatNumber(orderedTokens.length)} byte-level ranks.`],
    id,
    name: vocabulary.name,
    source: `Hugging Face: ${vocabulary.huggingfaceRepo}`,
  }
}
function getSpecialTokensFromConfig(tokenizerConfig: TokenizerConfigJson) {
  return Object.fromEntries(Object.entries(tokenizerConfig.added_tokens_decoder ?? {})
    .filter(([, token]) => token.special)
    .map(([id, token]) => [token.content, Number(id)]))
}
function buildTiktokenModelEncoder(id: string, vocabulary: VocabularyInfo, modelText: string, tokenizerConfig: TokenizerConfigJson): TokenEncoder {
  const ranks = modelText
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      const [token, rank] = line.split(/\s+/)
      return `! ${rank} ${token}`
    })
    .join('\n')
  const tiktoken = new Tiktoken({
    bpe_ranks: ranks,
    pat_str: String.raw`[\p{Script=Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
    special_tokens: getSpecialTokensFromConfig(tokenizerConfig),
  })
  return {
    countTokens: text => tiktoken.encode(text, [], []).length,
    details: ['Hugging Face tiktoken.model loaded through js-tiktoken.', `${formatNumber(modelText.split('\n').filter(Boolean).length)} ranks.`],
    id,
    name: vocabulary.name,
    source: `Hugging Face: ${vocabulary.huggingfaceRepo}`,
  }
}
async function writeTextFile(file: string, content: string) {
  await mkdir(path.dirname(file), {recursive: true})
  await writeFile(file, content)
}
async function getCachedHuggingFaceFile(repo: string, file: string) {
  const cacheFile = `${huggingFaceCacheRoot}/${repo.replaceAll('/', '--')}/${file}`
  try {
    return await readFile(cacheFile, 'utf8')
  } catch (error) {
    if (!(error instanceof Error) || !('code' in error) || error.code !== 'ENOENT') {
      throw error
    }
  }
  const response = await fetch(`https://huggingface.co/${repo}/resolve/main/${file}`)
  if (!response.ok) {
    throw new Error(`Could not fetch ${repo}/${file}: HTTP ${response.status}.`)
  }
  const text = await response.text()
  await writeTextFile(cacheFile, text)
  return text
}
async function loadVocabulary(id: string, vocabulary: VocabularyInfo): Promise<TokenEncoder> {
  if (vocabulary.import) {
    const ranks = (await import(vocabulary.import) as {default: ConstructorParameters<typeof Tiktoken>[0]}).default
    const tiktoken = new Tiktoken(ranks)
    return {
      countTokens: text => tiktoken.encode(text, [], []).length,
      details: ['Native js-tiktoken vocabulary export.'],
      id,
      name: vocabulary.name,
      source: vocabulary.import,
    }
  }
  if (vocabulary.huggingfaceRepo) {
    const tokenizerConfig = JSON.parse(await getCachedHuggingFaceFile(vocabulary.huggingfaceRepo, 'tokenizer_config.json')) as TokenizerConfigJson
    let tokenizerJsonText: string | undefined
    try {
      tokenizerJsonText = await getCachedHuggingFaceFile(vocabulary.huggingfaceRepo, 'tokenizer.json')
    } catch (error) {
      if (!(error instanceof Error) || !error.message.includes('HTTP 404')) {
        throw error
      }
    }
    if (tokenizerJsonText) {
      const tokenizerJson = JSON.parse(tokenizerJsonText) as TokenizerJson
      if (tokenizerJson.model?.type !== 'BPE') {
        throw new Error(`Unsupported Hugging Face tokenizer model type: ${tokenizerJson.model?.type ?? 'unknown'}.`)
      }
      if (hasByteLevelPreTokenizer(tokenizerJson.pre_tokenizer)) {
        return buildByteLevelTiktokenEncoder(id, vocabulary, tokenizerJson)
      }
      return new HuggingFaceBpeFallbackEncoder(id, vocabulary, tokenizerJson)
    }
    const modelText = await getCachedHuggingFaceFile(vocabulary.huggingfaceRepo, 'tiktoken.model')
    return buildTiktokenModelEncoder(id, vocabulary, modelText, tokenizerConfig)
  }
  throw new Error('Vocabulary must define either “import” or “huggingfaceRepo”.')
}
async function loadVocabularies() {
  const vocabularyMap = parseYaml(await readFile('src/vocabularies.yml', 'utf8')) as VocabularyMap
  const reports: Array<VocabularyReport> = []
  const encoders: Array<TokenEncoder> = []
  for (const [id, vocabulary] of Object.entries(vocabularyMap)) {
    try {
      const encoder = await loadVocabulary(id, vocabulary)
      encoders.push(encoder)
      reports.push({
        details: encoder.details,
        id,
        name: encoder.name,
        source: encoder.source,
      })
    } catch (error) {
      reports.push({
        details: [],
        error: error instanceof Error ? error.message : String(error),
        id,
        name: vocabulary.name,
        source: vocabulary.import ?? `Hugging Face: ${vocabulary.huggingfaceRepo}`,
      })
    }
  }
  return {
    encoders,
    reports,
  }
}
async function loadFixture(folder: string, encoders: Array<TokenEncoder>): Promise<FixtureReport> {
  const info = parseYaml(await readFile(`${folder}/info.yml`, 'utf8')) as FixtureInfo
  if (!Array.isArray(info.candidates)) {
    throw new TypeError(`${folder}/info.yml must contain a candidates array.`)
  }
  const sentenceReports = info.candidates.map(candidate => {
    const text = resolveCandidateText(info, candidate)
    const counts = Object.fromEntries(encoders.map(encoder => [encoder.id, encoder.countTokens(text)]))
    const values = Object.values(counts).filter(value => value !== undefined)
    return {
      candidate: candidateToString(candidate),
      counts,
      minimum: values.length ? Math.min(...values) : undefined,
      text,
    }
  })
  const folderName = folder.split('/').at(-1)!
  const totals = Object.fromEntries(encoders.map(encoder => [
    encoder.id,
    sentenceReports.reduce((sum, sentence) => sum + (sentence.counts[encoder.id] ?? 0), 0),
  ]))
  return {
    folderName,
    sentenceReports,
    title: getFixtureTitle(folderName),
    totals,
  }
}
class HuggingFaceBpeFallbackEncoder implements TokenEncoder {
  details: Array<string>
  id: string
  mergeRanks: Map<string, number>
  name: string
  source: string
  tokenizerJson: TokenizerJson
  vocab: Record<string, number>

  constructor(id: string, vocabulary: VocabularyInfo, tokenizerJson: TokenizerJson) {
    const vocab = tokenizerJson.model?.vocab
    const merges = tokenizerJson.model?.merges
    if (!vocab || !merges) {
      throw new Error('Hugging Face tokenizer.json does not contain model.vocab and model.merges.')
    }
    this.id = id
    this.name = vocabulary.name
    this.source = `Hugging Face: ${vocabulary.huggingfaceRepo}`
    this.tokenizerJson = tokenizerJson
    this.vocab = vocab
    this.mergeRanks = new Map(merges.map((merge, index) => {
      const [left, right] = Array.isArray(merge) ? merge : merge.split(' ')
      return [this.getMergeKey(left, right), index]
    }))
    this.details = ['Hugging Face BPE fallback used for a non-ByteLevel tokenizer because js-tiktoken is byte-BPE-only.', `${formatNumber(this.mergeRanks.size)} merge rules.`]
  }

  countTokens(text: string) {
    const normalized = applyNormalizer(text, this.tokenizerJson.normalizer)
    return this.getPretokenizedPieces(normalized).reduce((sum, piece) => sum + this.encodePiece(piece).length, 0)
  }

  encodePiece(piece: string) {
    const parts = [...piece].flatMap(char => (this.vocab[char] === undefined ? this.getByteFallbackTokens(char) : [char]))
    while (parts.length > 1) {
      let bestIndex = -1
      let bestRank = Number.POSITIVE_INFINITY
      for (let index = 0; index < parts.length - 1; index++) {
        const rank = this.mergeRanks.get(this.getMergeKey(parts[index], parts[index + 1]))
        if (rank !== undefined && rank < bestRank) {
          bestIndex = index
          bestRank = rank
        }
      }
      if (bestIndex === -1) {
        break
      }
      parts.splice(bestIndex, 2, parts[bestIndex] + parts[bestIndex + 1])
    }
    return parts
  }

  getByteFallbackTokens(char: string) {
    return [...textEncoder.encode(char)].map(byte => `<0x${byte.toString(16).padStart(2, '0').toUpperCase()}>`)
  }

  getMergeKey(left: string, right: string) {
    return `${left}\u0000${right}`
  }

  getPretokenizedPieces(text: string) {
    return text ? [text] : []
  }
}
await rm('out', {
  force: true,
  recursive: true,
})
await mkdir('out', {recursive: true})
const {encoders} = await loadVocabularies()
const fixtureFolders = (await globby('src/fixture/*', {onlyDirectories: true})).toSorted((a, b) => a.localeCompare(b))
const fixtures = await Promise.all(fixtureFolders.map(folder => loadFixture(folder, encoders)))
const results = fixtures.map(fixture => ({
  name: fixture.title,
  folderName: fixture.folderName,
  candidates: fixture.sentenceReports.map(sentence => ({
    name: sentence.candidate,
    text: sentence.text,
    scores: sentence.counts,
  })),
}))
await writeTextFile('out/result.yml', stringifyYaml(results))
console.log(`Updated token scores for ${formatNumber(fixtures.length)} fixtures in out/result.yml.`)
