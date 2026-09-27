// Ejecutar desde cualquier directorio: node src/05_duplicates_development.cjs
// Auditoría descriptiva; solo lee DEVELOPMENT y no elimina registros.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const source = path.join(root, 'data/splits/development_80.csv');
const digest = data => crypto.createHash('sha256').update(data).digest('hex');
const input = fs.readFileSync(source);
const lines = input.toString('utf8').trim().split(/\r?\n/);
const columns = lines.shift().split(',');
const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT'];
const fields = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'];
const start = Date.parse('2020-08-11T06:00:00Z');
const end = Date.parse('2025-07-01T18:00:00Z');
const groups = Object.fromEntries(symbols.map(s => [s, []]));
const exactSeen = new Set();
const keySeen = new Set();
let exactDuplicates = 0, duplicateKeys = 0;
for (const line of lines) {
  const values = line.split(',');
  if (values.length !== columns.length) throw Error('Estructura CSV inesperada');
  const row = Object.fromEntries(columns.map((c, i) => [c, values[i]]));
  const t = Date.parse(row.open_time);
  if (!groups[row.symbol] || !Number.isFinite(t) || t < start || t > end || (t-start)%3600000) throw Error('Registro fuera de DEVELOPMENT');
  const key = `${row.symbol}|${t}`;
  if (exactSeen.has(line)) exactDuplicates++;
  if (keySeen.has(key)) duplicateKeys++;
  exactSeen.add(line); keySeen.add(key);
  for (const field of [...fields, 'number_of_trades']) {
    if (!row[field]?.trim() || !Number.isFinite(Number(row[field]))) throw Error(`Valor no numérico: ${field}`);
  }
  groups[row.symbol].push({t, row});
}
// Umbral descriptivo fijo: diferencia relativa simétrica <= 0.000001
// en TODOS los 8 campos; number_of_trades debe ser exactamente igual.
const tolerance = 1e-6;
const relativeDifference = (a, b) => a === b ? 0 : Math.abs(a-b) / Math.max(Math.abs(a), Math.abs(b));
const candidates = [], bySymbol = [];
for (const symbol of symbols) {
  const rows = groups[symbol].sort((a,b) => a.t-b.t);
  let pairs = 0, matches = 0;
  for (let i=1; i<rows.length; i++) {
    const a = rows[i-1], b = rows[i];
    if (b.t-a.t !== 3600000) continue; // No cruzar huecos.
    pairs++;
    const differences = fields.map(f => relativeDifference(Number(a.row[f]), Number(b.row[f])));
    if (Number(a.row.number_of_trades) === Number(b.row.number_of_trades) && differences.every(d => d <= tolerance)) {
      matches++;
      candidates.push({symbol, first:a.row.open_time, second:b.row.open_time, max_relative_difference:Math.max(...differences)});
    }
  }
  bySymbol.push({symbol, rows:rows.length, eligible_pairs:pairs, candidates:matches});
}
const report = {source:'data/splits/development_80.csv', sha256:digest(input),
  rows:lines.length, exact_duplicates:exactDuplicates, duplicate_symbol_time:duplicateKeys,
  scope:'Pares del mismo activo separados exactamente por una hora; no se comparan horas no consecutivas ni activos diferentes.',
  fields, tolerance, number_of_trades:'igualdad numérica exacta', by_symbol:bySymbol, candidates};
if (digest(fs.readFileSync(source)) !== report.sha256) throw Error('DEVELOPMENT cambió');
fs.mkdirSync(path.join(root, 'outputs/tables'), {recursive:true});
fs.writeFileSync(path.join(root, 'outputs/tables/development_duplicates.json'), JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
