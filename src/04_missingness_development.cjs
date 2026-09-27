// Ejecutar: node src/04_missingness_development.cjs
// Solo lee DEVELOPMENT; no imputa, modifica datos ni construye el objetivo.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const source = path.join(root, 'data/splits/development_80.csv');
const raw = fs.readFileSync(source);
const hash = b => crypto.createHash('sha256').update(b).digest('hex');
const lines = raw.toString('utf8').trim().split(/\r?\n/);
const columns = lines.shift().split(',');
const symbols = ['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','SOLUSDT'];
const start = Date.parse('2020-08-11T06:00:00Z');
const end = Date.parse('2025-07-01T18:00:00Z');
const hour = 3600000;
const n = (end-start)/hour+1;
const maps = Object.fromEntries(symbols.map(s=>[s,new Map()]));
const nulls = Object.fromEntries(symbols.map(s=>[s,columns.map(()=>0)]));
for (const line of lines) {
  const values = line.split(',');
  if(values.length !== columns.length) throw Error('Estructura CSV inesperada');
  const row = Object.fromEntries(columns.map((c,i)=>[c,values[i]]));
  const t = Date.parse(row.open_time);
  if(!maps[row.symbol] || !Number.isFinite(t) || t<start || t>end || (t-start)%hour) throw Error('Registro fuera de DEVELOPMENT');
  if(maps[row.symbol].has(t)) throw Error('Timestamp duplicado');
  maps[row.symbol].set(t,row);
  values.forEach((v,i)=>{if(/^(?:|na|nan|null|none|nat)$/i.test(v.trim())) nulls[row.symbol][i]++;});
}
const iso = t=>new Date(t).toISOString().replace('.000Z',' UTC').replace('T',' ');
const missing = Object.fromEntries(symbols.map(s=>[s,Array.from({length:n},(_,i)=>start+i*hour).filter(t=>!maps[s].has(t))]));
const common = missing[symbols[0]];
if(!symbols.every(s=>JSON.stringify(missing[s])===JSON.stringify(common))) throw Error('Patrones distintos: adaptar figura');
const gaps=[];
for(const t of common){const last=gaps.at(-1);if(last && t===last.end+hour){last.end=t;last.hours++;}else gaps.push({start:t,end:t,hours:1});}
const median = a=>{if(!a.length)return null;const b=[...a].sort((x,y)=>x-y);const k=Math.floor(b.length/2);return b.length%2?b[k]:(b[k-1]+b[k])/2;};
const comparison = symbols.map(s=>{
  const groups={gap:[],observed:[]};
  for(let t=start+2*hour;t<=end;t+=hour){
    const a=maps[s].get(t-hour),b=maps[s].get(t-2*hour);
    if(!a||!b)continue;
    const x=Number(a.close),y=Number(b.close);
    if(!(x>0&&y>0))throw Error('Precio no válido para retorno previo');
    groups[maps[s].has(t)?'observed':'gap'].push(100*Math.abs(Math.log(x/y)));
  }
  return {symbol:s,gap_n:groups.gap.length,observed_n:groups.observed.length,gap_median:median(groups.gap),observed_median:median(groups.observed)};
});
const report={source:'data/splits/development_80.csv',sha256:hash(raw),rows:lines.length,columns,expected_per_asset:n,nulls,coverage:symbols.map(s=>({symbol:s,observed:maps[s].size,missing:missing[s].length,percent:100*missing[s].length/n})),gaps:gaps.map(g=>({start:iso(g.start),end:iso(g.end),hours:g.hours})),comparison};
fs.mkdirSync(path.join(root,'outputs/tables'),{recursive:true});
fs.writeFileSync(path.join(root,'outputs/tables/development_missingness.json'),JSON.stringify(report,null,2)+'\n');
// Matriz binaria: cada fila representa exactamente una hora ausente;
// cada columna, una variable de mercado. El patrón se verifica para los 5 activos.
const fields=columns.filter(c=>!['open_time','symbol'].includes(c));
const width=1200,height=210+common.length*25;
const parts=[`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" fill="#172b4d"><text x="25" y="30" font-size="22">Matriz de faltantes — DEVELOPMENT</text><text x="25" y="56" font-size="14">Detalle de horas ausentes; patrón idéntico en BTC, ETH, BNB, XRP y SOL</text>`];
fields.forEach((c,i)=>parts.push(`<text x="${280+i*88}" y="135" font-size="11" transform="rotate(-24 ${280+i*88} 135)">${c}</text>`));
common.forEach((t,j)=>{const y=150+j*25;parts.push(`<text x="25" y="${y+17}" font-size="13">${iso(t)}</text>`);fields.forEach((_,i)=>parts.push(`<rect x="${280+i*88}" y="${y}" width="85" height="22" fill="#c2410c"/>`));});
parts.push(`<text x="25" y="${height-26}" font-size="14">Naranja: ausente. Horas observadas por activo: ${maps[symbols[0]].size}; celdas nulas en ellas: ${Object.values(nulls).flat().reduce((a,b)=>a+b,0)}.</text></g></svg>`);
fs.mkdirSync(path.join(root,'book/_static/figures'),{recursive:true});
fs.writeFileSync(path.join(root,'book/_static/figures/development_missing_matrix.svg'),parts.join('\n'));
if(hash(fs.readFileSync(source))!==report.sha256)throw Error('DEVELOPMENT cambió durante la ejecución');
console.log(JSON.stringify(report,null,2));
