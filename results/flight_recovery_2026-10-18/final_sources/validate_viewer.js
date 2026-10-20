const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
 const root=process.argv[2],browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:1280,height:1150}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8792/viewer.html');await page.waitForFunction(()=>!!window.flightViewer);
 const cases=await page.locator('#trial option').evaluateAll(xs=>xs.map(x=>x.value)),checks=[];
 for(const name of cases){
  await page.selectOption('#trial',name);const before=await page.evaluate(()=>flightViewer.scene.getObjectByName('fixed_marker').position.toArray());
  await page.locator('#time').evaluate(x=>{x.value=1000;x.dispatchEvent(new Event('input'))});
  checks.push(await page.evaluate(({before,name})=>{
   const v=flightViewer,r=v.row,point=v.scene.getObjectByName('position_point'),p=point.position.toArray(),expected=r.p.at(-1),names=v.scene.children.map(x=>x.name);
   return {name,traceMatches:Math.max(Math.abs(p[0]-expected[0]),Math.abs(p[1]-expected[2]),Math.abs(p[2]+expected[1]))<1e-12,targetFixed:JSON.stringify(before)===JSON.stringify(v.scene.getObjectByName('fixed_marker').position.toArray()),markerVisibility:v.scene.getObjectByName('fixed_marker').visible===!r.absent,bodyHidden:names.every(n=>!n||['fixed_marker','position_point','measured_path'].includes(n))};
  },{before,name}));
 }
 await page.selectOption('#trial',cases[0]);await page.click('#play');await page.waitForTimeout(350);const advanced=await page.evaluate(()=>flightViewer.index>0);await page.click('#play');const stopped=await page.evaluate(()=>flightViewer.index);await page.waitForTimeout(150);const paused=await page.evaluate(i=>flightViewer.index===i,stopped);await page.click('#reset');const reset=await page.evaluate(()=>flightViewer.index===0);
 await page.locator('#time').evaluate(x=>{x.value=1000;x.dispatchEvent(new Event('input'))});await page.screenshot({path:root+'/preview.png',fullPage:true});
 const result={cases:checks,errors,advanced,paused,reset,all_pass:cases.length===5&&!errors.length&&advanced&&paused&&reset&&checks.every(r=>r.traceMatches&&r.targetFixed&&r.markerVisibility&&r.bodyHidden)};
 fs.writeFileSync(root+'/browser_validation.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));await browser.close();if(!result.all_pass)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
