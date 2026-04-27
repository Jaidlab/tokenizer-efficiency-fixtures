import {mkdir, readFile, rm, writeFile} from 'node:fs/promises'
import * as path from 'node:path/posix'

import {parse as parseYaml} from 'yaml'

type VocabularyInfo = {
  huggingfaceRepo?: string
  import?: string
  name: string
}

type VocabularyMap = Record<string, VocabularyInfo>

type CandidateResult = {
  name: string
  scores: Record<string, number | undefined>
  text: string
}

type FixtureResult = {
  candidates: Array<CandidateResult>
  folderName: string
  name: string
}

type VocabularyReport = {
  id: string
  name: string
  source: string
}

type CandidateMetrics = CandidateResult & {
  average?: number
  rank: number
  total?: number
}

type FixtureReport = FixtureResult & {
  candidateMetrics: Array<CandidateMetrics>
  vocabularyWinners: Record<string, number | undefined>
  winningCandidates: Array<CandidateMetrics>
}

const pageRoot = 'dist/page'
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
function formatAverage(value: number | undefined) {
  if (value === undefined) {
    return '–'
  }
  return Number.isInteger(value) ? formatNumber(value) : value.toFixed(1)
}
function getVocabularySource(vocabulary: VocabularyInfo) {
  return vocabulary.import ?? `Hugging Face: ${vocabulary.huggingfaceRepo}`
}
function getScoreValues(candidate: CandidateResult, vocabularyReports: Array<VocabularyReport>) {
  return vocabularyReports
    .map(vocabulary => candidate.scores[vocabulary.id])
    .filter(score => score !== undefined)
}
function getCandidateTotal(candidate: CandidateResult, vocabularyReports: Array<VocabularyReport>) {
  const values = getScoreValues(candidate, vocabularyReports)
  if (!values.length) {
    return
  }
  return values.reduce((sum, value) => sum + value, 0)
}
function getCandidateAverage(candidate: CandidateResult, vocabularyReports: Array<VocabularyReport>) {
  const values = getScoreValues(candidate, vocabularyReports)
  if (!values.length) {
    return
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length
}
function getCandidateSortValue(candidate: CandidateMetrics) {
  return candidate.total ?? Number.POSITIVE_INFINITY
}
function getVocabularyWinners(candidates: Array<CandidateResult>, vocabularyReports: Array<VocabularyReport>) {
  return Object.fromEntries(vocabularyReports.map(vocabulary => {
    const scores = candidates.map(candidate => candidate.scores[vocabulary.id]).filter(score => score !== undefined)
    return [vocabulary.id, scores.length ? Math.min(...scores) : undefined]
  }))
}
function makeFixtureReport(fixture: FixtureResult, vocabularyReports: Array<VocabularyReport>): FixtureReport {
  const sortedCandidates = fixture.candidates
    .map(candidate => ({
      ...candidate,
      average: getCandidateAverage(candidate, vocabularyReports),
      rank: 0,
      total: getCandidateTotal(candidate, vocabularyReports),
    }))
    .toSorted((a, b) => getCandidateSortValue(a) - getCandidateSortValue(b) || a.name.localeCompare(b.name))
  let previousScore: number | undefined
  let previousRank = 0
  const candidateMetrics = sortedCandidates.map((candidate, index) => {
    const score = candidate.total
    const rank = score === previousScore ? previousRank : index + 1
    previousScore = score
    previousRank = rank
    return {
      ...candidate,
      rank,
    }
  })
  const bestScore = candidateMetrics[0]?.total
  return {
    ...fixture,
    candidateMetrics,
    vocabularyWinners: getVocabularyWinners(fixture.candidates, vocabularyReports),
    winningCandidates: bestScore === undefined ? [] : candidateMetrics.filter(candidate => candidate.total === bestScore),
  }
}
async function writeTextFile(file: string, content: string) {
  await mkdir(path.dirname(file), {recursive: true})
  await writeFile(file, content)
}
function renderPage(title: string, body: string) {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${html(title)}</title>
  <style>${css}</style>
</head>
<body>
  <main id="root">
${body}
  </main>
</body>
</html>
`
}
function renderVocabularyBadges(vocabularyReports: Array<VocabularyReport>) {
  return vocabularyReports.map(vocabulary => `<span class="badge blue" title="${attribute(vocabulary.source)}">${html(vocabulary.name)}</span>`).join('\n')
}
function renderCandidateLabel(candidate: CandidateResult) {
  return `<code>${html(candidate.name)}</code>`
}
function renderFixtureReport(fixture: FixtureReport, vocabularyReports: Array<VocabularyReport>) {
  const body = `    <nav class="breadcrumb"><a href="../">all fixtures</a> / <span>${html(fixture.name)}</span></nav>
    <header class="hero">
      <h1>${html(fixture.name)}</h1>
      <p class="subtitle">Candidate sentences ranked by total token cost across ${formatNumber(vocabularyReports.length)} vocabularies. Lower is better.</p>
      <div class="badges">${renderVocabularyBadges(vocabularyReports)}</div>
    </header>
    <section class="winner-grid">
      ${fixture.winningCandidates.map(candidate => `<article class="card winner-card">
        <div class="label">winning candidate</div>
        <h2>${renderCandidateLabel(candidate)}</h2>
        <pre class="winner-text">${html(candidate.text)}</pre>
        <div class="stat-row"><span><strong>${formatNumber(candidate.total)}</strong> total tokens</span><span><strong>${formatAverage(candidate.average)}</strong> avg</span></div>
      </article>`).join('\n')}
    </section>
    <section class="card table-card">
      <h2>Candidate ranking</h2>
      <table>
        <thead><tr><th>rank</th><th>candidate</th><th>sentence</th><th>total</th><th>avg</th>${vocabularyReports.map(vocabulary => `<th>${html(vocabulary.name)}</th>`).join('')}</tr></thead>
        <tbody>
          ${fixture.candidateMetrics.map(candidate => `<tr class="${candidate.rank === 1 ? 'winner-row' : ''}">
            <td class="rank">#${formatNumber(candidate.rank)}</td>
            <td>${renderCandidateLabel(candidate)}</td>
            <td><pre class="sentence">${html(candidate.text)}</pre></td>
            <td class="number total ${candidate.rank === 1 ? 'winner' : ''}">${formatNumber(candidate.total)}</td>
            <td class="number">${formatAverage(candidate.average)}</td>
            ${vocabularyReports.map(vocabulary => {
              const score = candidate.scores[vocabulary.id]
              return `<td class="number ${score !== undefined && score === fixture.vocabularyWinners[vocabulary.id] ? 'column-winner' : ''}">${formatNumber(score)}</td>`
            }).join('')}
          </tr>`).join('\n')}
        </tbody>
      </table>
    </section>
    <section class="card">
      <h2>Vocabularies</h2>
      <div class="vocabulary-grid">${vocabularyReports.map(vocabulary => `<article><h3>${html(vocabulary.name)}</h3><p class="meta">${html(vocabulary.source)}</p></article>`).join('\n')}</div>
    </section>`
  return renderPage(`${fixture.name} · tokenizer efficiency`, body)
}
function renderIndex(fixtures: Array<FixtureReport>, vocabularyReports: Array<VocabularyReport>) {
  const body = `    <header class="hero">
      <h1>Tokenizer efficiency fixtures</h1>
      <p class="subtitle">Static reports for picking the shortest text candidate across tokenizer vocabularies.</p>
      <div class="badges">${renderVocabularyBadges(vocabularyReports)}</div>
    </header>
    <section class="fixture-grid">
      ${fixtures.map(fixture => `<article class="card fixture-card">
        <h2><a href="${attribute(encodeURI(`${fixture.folderName}/`))}">${html(fixture.name)}</a></h2>
        <p class="meta">${formatNumber(fixture.candidates.length)} candidates · ${formatNumber(vocabularyReports.length)} vocabularies</p>
        <div class="fixture-winner">
          <span class="label">winner</span>
          ${fixture.winningCandidates.map(renderCandidateLabel).join(' ')}
        </div>
        <table>
          <thead><tr><th>rank</th><th>candidate</th><th>total</th></tr></thead>
          <tbody>${fixture.candidateMetrics.map(candidate => `<tr><td class="rank">#${formatNumber(candidate.rank)}</td><td>${renderCandidateLabel(candidate)}</td><td class="number ${candidate.rank === 1 ? 'winner' : ''}">${formatNumber(candidate.total)}</td></tr>`).join('\n')}</tbody>
        </table>
      </article>`).join('\n')}
    </section>`
  return renderPage('Tokenizer efficiency fixtures', body)
}
const css = String.raw`
* { box-sizing: border-box; margin: 0; padding: 0; }
:root { color-scheme: dark; --background: #000; --panel: #111; --panel-strong: #171717; --border: #242424; --text: #fff; --weak: #b8b8b8; --primary: hsl(330 100% 60%); --green: #80ff33; --blue: #3399ff; --yellow: #ffff33; --text-font: Geologica, Rubik, Calibri, sans-serif; --ui-font: Arial, sans-serif; --code-font: JetBrainsMono NF, JetBrains Mono, Symbols Nerd Font Mono, Symbols Nerd Font Mono Regular, monospace; }
body { min-height: 100vh; background: var(--background); color: var(--text); font-family: var(--text-font); line-height: 1.5; }
#root { max-width: 1280px; min-height: 100vh; margin: 0 auto; padding: 24px; }
a { color: var(--primary); text-decoration: none; }
a:hover { text-decoration: underline; }
h1, h2, h3 { font-family: var(--ui-font); font-weight: 600; line-height: 1.2; }
h1 { font-size: 1.55rem; }
h2 { font-size: 1.15rem; margin-bottom: 12px; }
h3 { font-size: 1rem; }
.hero { margin-bottom: 24px; }
.subtitle { color: var(--weak); font-size: .9rem; margin-top: 6px; }
.breadcrumb { color: var(--weak); font-family: var(--ui-font); font-size: .85rem; margin-bottom: 18px; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.badge { display: inline-block; border-radius: 12px; font-family: var(--ui-font); font-size: .75rem; font-weight: 500; padding: 2px 8px; }
.badge.blue { background: #001a33; color: var(--blue); }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.fixture-grid, .vocabulary-grid, .winner-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); margin-bottom: 24px; }
.winner-card { border-color: color-mix(in srgb, var(--green) 50%, var(--border)); box-shadow: 0 0 24px rgb(128 255 51 / 8%); }
.label { color: var(--weak); display: block; font-family: var(--ui-font); font-size: .78rem; margin-bottom: 6px; text-transform: lowercase; }
.meta { color: var(--weak); font-size: .85rem; margin-top: 4px; }
.fixture-winner { margin: 14px 0; }
.stat-row { color: var(--weak); display: flex; flex-wrap: wrap; gap: 14px; margin-top: 12px; }
.stat-row strong { color: var(--green); font-family: var(--code-font); }
.table-card { margin-bottom: 24px; overflow-x: auto; }
table { border-collapse: collapse; font-family: var(--code-font); font-size: .8rem; margin-top: 8px; width: 100%; }
th { background: var(--panel-strong); border-bottom: 2px solid var(--border); color: var(--weak); font-family: var(--ui-font); font-size: .85rem; padding: 8px 12px; text-align: left; white-space: nowrap; }
td { border-bottom: 1px solid var(--border); padding: 8px 12px; vertical-align: top; }
tbody tr:hover { background: rgb(255 255 255 / 3%); }
.winner-row { background: rgb(128 255 51 / 4%); }
code, pre { font-family: var(--code-font); }
code { color: var(--yellow); overflow-wrap: anywhere; }
.sentence, .winner-text { white-space: pre-wrap; }
.sentence { max-width: 560px; }
.winner-text { color: var(--weak); margin-top: 8px; }
.number, .rank { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.rank { color: var(--weak); }
.winner { color: var(--green); font-weight: 700; }
.column-winner { color: var(--blue); }
.total { font-size: .9rem; }
.fixture-card table th, .fixture-card table td { padding: 5px 8px; }
.fixture-card table th:first-child, .fixture-card table td:first-child { padding-left: 0; }
.fixture-card table th:last-child, .fixture-card table td:last-child { padding-right: 0; }
`
const vocabularyMap = parseYaml(await readFile('src/vocabularies.yml', 'utf8')) as VocabularyMap
const vocabularyReports = Object.entries(vocabularyMap).map(([id, vocabulary]) => ({
  id,
  name: vocabulary.name,
  source: getVocabularySource(vocabulary),
}))
const result = parseYaml(await readFile('out/result.yml', 'utf8')) as Array<FixtureResult>
const fixtures = result.map(fixture => makeFixtureReport(fixture, vocabularyReports))
await rm(pageRoot, {
  force: true,
  recursive: true,
})
await mkdir(pageRoot, {recursive: true})
for (const fixture of fixtures) {
  await writeTextFile(`${pageRoot}/${fixture.folderName}/index.html`, renderFixtureReport(fixture, vocabularyReports))
}
await writeTextFile(`${pageRoot}/index.html`, renderIndex(fixtures, vocabularyReports))
console.log(`Built ${formatNumber(fixtures.length)} fixture reports in ${pageRoot}.`)
