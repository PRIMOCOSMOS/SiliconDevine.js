"""Build the launcher help from the one maintained user guide (dev dependency: markdown)."""
from pathlib import Path
import re
try:
    import markdown
except ImportError:
    raise SystemExit('Install the documentation build tool: python -m pip install markdown')
root=Path(__file__).resolve().parents[1]
source=(root/'docs/USER_GUIDE.md').read_text(encoding='utf-8-sig')
body=markdown.markdown(source,extensions=['fenced_code','tables','toc'])
body=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',body,flags=re.S)
style='''body{margin:0;background:#080c12;color:#eef3f8;font:16px/1.8 "Segoe UI","Microsoft YaHei UI",sans-serif}main{max-width:920px;margin:48px auto;padding:0 24px 80px}h1{font-size:36px;line-height:1.3}h2{font-size:23px;margin:48px 0 16px;scroll-margin-top:20px}h3{font-size:19px;margin-top:30px}p,li{color:#d0dce6}a{color:#a9d9f7}pre{padding:20px;background:#151f2b;border-radius:12px;overflow:auto;font:14px/1.7 Consolas,monospace}code{overflow-wrap:anywhere}pre code{overflow-wrap:normal}table{border-collapse:collapse;width:100%;min-width:460px;font-size:14px}td,th{padding:12px;text-align:left;border-bottom:1px solid #344452;vertical-align:top}.table-scroll{overflow:auto}nav{padding:18px 0;border-bottom:1px solid #344452}nav a{margin-right:18px;display:inline-block}a:focus-visible{outline:2px solid #b5e2ff;outline-offset:4px}@media(max-width:600px){main{padding:0 16px 48px;margin-top:24px}h1{font-size:28px}h2{font-size:21px}pre{padding:14px}}'''
nav='<nav aria-label="文档导航"><a href="INDEX.html">参考目录</a><a href="#2">代码接入</a><a href="#6">看图与操作</a><a href="#8">常见问题</a></nav>'
for page in (root/'docs').glob('*.md'):
    text=page.read_text(encoding='utf-8-sig')
    content=markdown.markdown(text,extensions=['fenced_code','tables','toc'])
    content=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',content,flags=re.S)
    def local_link(match):
        stem=match.group(1)
        if not (root/'docs'/f'{stem}.md').is_file():return match.group(0)
        return 'href="'+('QUICKSTART' if stem=='USER_GUIDE' else stem)+'.html"'
    content=re.sub(r'href="([^"/:]+)\.md"',local_link,content)
    target=page.with_name('QUICKSTART.html') if page.stem=='USER_GUIDE' else page.with_suffix('.html')
    title=text.splitlines()[0].lstrip('# ')
    navigation=nav if page.stem=='USER_GUIDE' else '<nav><a href="QUICKSTART.html">使用说明</a><a href="INDEX.html">文档目录</a></nav>'
    target.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+style+'</style></head><body><main>'+navigation+content+'</main></body></html>',encoding='utf-8')
print('Generated browser documentation from the maintained Markdown sources')
