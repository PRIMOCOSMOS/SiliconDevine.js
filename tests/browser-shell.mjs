// Follow the public shell navigation so renderer regressions work with lazy panels.
export async function selectExample(page,value){
 if(!await page.locator('#model-library').isVisible())await page.click('#catalog-toggle');
 await page.selectOption('#example',value);
 await page.click('#library-return');
}
export async function selectModule(page,value){
 if(!await page.locator('#model-inspector').isVisible())await page.click('#inspector-toggle');
 await page.selectOption('#modules',value);
 await page.click('#inspector-close');
}
