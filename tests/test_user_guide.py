"""Check that the first runnable guide example works with every documented backend."""
import unittest,re
from pathlib import Path
from silicondevine import export_model

class UserGuideTest(unittest.TestCase):
    def test_model_example(self):
        text=(Path(__file__).resolve().parents[1]/'docs/USER_GUIDE.md').read_text(encoding='utf-8-sig')
        code=re.search(r'```python\n(.*?)```',text,re.S).group(1)
        scope={};exec(compile(code,'USER_GUIDE.md','exec'),scope)
        model,args=scope['build_model']()
        for backend in ['fx','export','execution']:
            with self.subTest(backend=backend):
                graph=export_model(model,args,backend=backend,include_values=True)
                out=next(t for t in graph['tensors'] if t['id']==graph['outputs'][0])
                self.assertEqual(out['shape'],[2,8])
                self.assertEqual(len(out['data']['values']),16)
    def test_html_navigation(self):
        docs=Path(__file__).resolve().parents[1]/'docs'
        for page in docs.glob('*.html'):
            html=page.read_text(encoding='utf-8')
            for url in re.findall(r'href="([^"]+)"',html):
                if url.startswith('#'):
                    self.assertIn('id="'+url[1:]+'"',html)
                elif not re.match(r'\w+://',url):
                    self.assertTrue((page.parent/url.split('#')[0]).exists(),f'{page.name}: {url}')

if __name__=='__main__':unittest.main()
