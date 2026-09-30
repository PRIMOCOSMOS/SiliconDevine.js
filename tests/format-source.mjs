import ts from 'typescript';
import {readdir,readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
const printer=ts.createPrinter({newLine:ts.NewLineKind.LineFeed});
async function format(dir){for(const file of await readdir(dir,{withFileTypes:true})){const path=join(dir,file.name);if(file.isDirectory())await format(path);else if(path.endsWith('.ts')){const source=ts.createSourceFile(path,await readFile(path,'utf8'),ts.ScriptTarget.Latest,true,ts.ScriptKind.TS);await writeFile(path,printer.printFile(source));}}}
await format('src');await format('demo');
