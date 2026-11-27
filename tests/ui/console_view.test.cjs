const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../../src/bucket_robot/runtime/index.html'), 'utf8');
const script = html.match(/<script id="console-view-model">([\s\S]*?)<\/script>/)[1];
const view = vm.runInNewContext(script + '; consoleView');
const video = { statusAgeSeconds: .01, videoAgeSeconds: .05, deliveredFPS: 16.8 };
function observation() {
  return {
    frame_id: 40, arrival: 100, published_at: 100.02, frame_age_s: .06,
    detected: true, distance_m: .48, distance_kind: 'camera_forward_Z_bucket_body',
    distance_age_ms: 50, distance_source_frame: 38, measurement_status: 'held',
    health: { components: Object.fromEntries(['camera', 'calibration', 'detector', 'ranging', 'serial'].map(key => [key, { state: 'ready' }])) },
    tracking: { target_age_s: .04, target: { offset_px: [-152.2, 84] },
      arm: { timestamp: 100.03, telemetry: { servo_software_deg: 73, lift_position_mm: 0, slide_position_mm: 10 } } },
  };
}
test('real forward Z remains in metres; current tracking offsets use right/down positive pixels', () => {
  const result = view.readings(observation(), video);
  assert.equal(result.distance, '0.48');
  assert.equal(result.offset, '−152 / +84');
  assert.equal(result.pose, 'A 73° · B 0.0 · C 10.0 mm');
  assert.equal(result.fps, '16.8');
});
test('temperatures expire after sensor age plus status delay; missing sensors stay unknown', () => {
  const thermal = { age_s: .8, sensors: [{ label: 'NPU', celsius: 63.769 }, { label: 'GPU', celsius: null }] };
  assert.equal(view.temperatures(thermal, .2)[0].value, '63.8');
  assert.equal(view.temperatures(thermal, .2)[1].value, '—');
  assert.equal(view.temperatures(thermal, 2.21)[0].value, '—');
  assert.equal(view.temperatures({ ...thermal, age_s: null }, 0)[0].value, '—');
  assert.equal(view.temperatures({ ...thermal, age_s: -1 }, 0)[0].value, '—');
});
test('missing data never becomes zero or a sample value', () => {
  const result = view.readings();
  for (const key of ['distance', 'offset', 'fps', 'pose']) assert.equal(result[key], '—');
});
test('model failure clears spatial readings while independently delivered video keeps its FPS', () => {
  const source = observation();
  source.health.components.detector.state = 'unavailable';
  source.tracking_fps = 0;
  const result = view.readings(source, video);
  assert.equal(result.distance, '—');
  assert.equal(result.offset, '—');
  assert.equal(result.fps, '16.8');
});
test('old status cannot keep readings alive when live video still arrives', () => {
  const result = view.readings(observation(), { ...video, statusAgeSeconds: 2 });
  assert.equal(result.fps, '16.8');
  assert.equal(result.distance, '—');
  assert.equal(result.offset, '—');
  assert.equal(result.pose, '—');
});
test('stale video reports no FPS independently of healthy model or tracking FPS', () => {
  const result = view.readings(observation(), { ...video, videoAgeSeconds: .401 });
  assert.equal(result.fps, '—');
  assert.equal(result.videoFresh, false);
});
test('measurement hold expiry includes time since metadata publication and status receipt', () => {
  const source = observation(); source.distance_age_ms = 1460;
  assert.equal(view.readings(source, video).distance, '—');
  source.distance_age_ms = 1400;
  assert.equal(view.readings(source, video).distance, '0.48');
});
test('missing units, invalid values, unavailable ranging, and inconsistent source frames are not measurements', () => {
  for (const fields of [{ distance_m: null }, { distance_m: '0.48' }, { distance_m: NaN },
    { distance_m: 0 }, { distance_age_ms: -1 }, { distance_source_frame: 41 },
    { distance_source_frame: null }, { distance_kind: 'euclidean_distance' }, { measurement_status: 'invalid' }]) {
    assert.equal(view.readings({ ...observation(), ...fields }, video).distance, '—', JSON.stringify(fields));
  }
  const source = observation(); source.health.components.ranging.state = 'unavailable';
  assert.equal(view.readings(source, video).distance, '—');
});
test('old tracking positions cannot borrow a new video frame timestamp', () => {
  for (const target_age_s of [.401, -1, null]) {
    const source = observation(); source.tracking.target_age_s = target_age_s;
    assert.equal(view.readings(source, video).offset, '—');
  }
  const source = observation(); source.frame_age_s = .401;
  assert.equal(view.readings(source, video).offset, '—');
});
test('old arm telemetry stays hidden even while controller health has newer valid frames', () => {
  const source = observation(); source.tracking.arm.timestamp = 99;
  assert.equal(view.readings(source, video).pose, '—');
});
test('the check list reflects real component states and does not invent independent motor detection', () => {
  const components = { serial: { state: 'unavailable' }, camera: { state: 'ready' } };
  assert.equal(view.check('serial', components.serial, components).result, 'fail');
  for (const key of ['arm', 'chassis']) {
    const result = view.check(key, { state: 'unknown' }, components);
    assert.equal(result.result, 'skip'); assert.equal(result.label, '无法确认');
    assert.equal(view.check(key, { state: 'ready' }, { serial: { state: 'ready' } }).label, '回传就绪');
  }
  assert.equal(view.check('arm', { state: 'unavailable' }, { serial: { state: 'ready' } }).result, 'fail');
});
test('waiting for camera data is distinct from a reported model failure', () => {
  const components = { camera: { state: 'unavailable' } };
  assert.equal(view.check('detector', { state: 'stale' }, components).result, 'wait');
  assert.equal(view.check('detector', { state: 'unavailable', code: 'MODEL_FAILED' }, components).result, 'fail');
  assert.equal(view.check('camera', { state: 'ready' }, components).result, 'ok');
});
test('approved prototype replay and scene simulation are absent from the production page', () => {
  assert.ok(!html.includes('id="scenario"'));
  assert.ok(!html.includes('重放自检'));
  assert.ok(!html.includes('data:image/'));
  for (const id of ['retry', 'auto', 'select', 'cancel', 'start', 'stop', 'isolated', 'selection']) {
    assert.ok(html.includes(`id="${id}"`), id);
  }
});
test('re-fetching one frame cannot inflate the actual video FPS', () => {
  let rate = view.frameRate({ id: null, times: [] }, '1', 0);
  rate = view.frameRate(rate, '1', 100);
  assert.equal(rate.fps, 0);
  rate = view.frameRate(rate, '2', 200);
  rate = view.frameRate(rate, '2', 250);
  assert.equal(rate.fps, 5);
  rate = view.frameRate(rate, '3', 400);
  assert.equal(rate.fps, 5);
  assert.equal(view.frameRate(rate, null, 450).fps, 0);
  assert.equal(view.frameRate(rate, '4', 2501).fps, 0);
});
test('expected startup preparation is WAIT while a reported controller fault remains FAIL', () => {
  const item = { state: 'unavailable', code: 'CONTROLLER_NOT_READY' };
  const components = { serial: { state: 'ready' } };
  assert.equal(view.check('arm', item, components, { phase: 'startup_waiting' }).result, 'wait');
  assert.equal(view.check('arm', item, components, { phase: 'startup_waiting', motionLatch: { code: 'CONTROLLER_FAULT' } }).result, 'fail');
});

test('mission step list uses real phases and stays in the runtime log', () => {
  const p = { running: true, phase: 'post_extending_C', stages: ['acquire','track','post_align','drain','post_action'],
    post: { initial_state: { slide_position_mm: 0 }, lateral_completion: 'timed_estimate_no_odometry', B_top_confirmed: true } };
  const progress = view.missionProgress(p, null);
  assert.equal(progress.headline, 'C 伸出');
  assert.equal(progress.rows[6].result, 'run');
  for (const index of [0,1,2,3,4,5]) assert.equal(progress.rows[index].result, 'ok');
  assert.equal(progress.rows[7].result, 'wait');
  const log = html.slice(html.indexOf('<details id="runtimeLog"'),html.indexOf('</aside>'));
  assert.ok(log.includes('id="missionSteps"'));
  assert.ok(log.indexOf('id="missionSteps"') < log.indexOf('id="eventLog"'));
});
test('occlusion waits for the same bucket without declaring a completed approach', () => {
  const p = { running:true, phase:'waiting_target', stages:['acquire','track'] };
  const progress = view.missionProgress(p, null);
  assert.equal(progress.rows[0].result,'ok');
  assert.equal(progress.rows[1].result,'wait');
  assert.match(progress.detail,/同一目标/);
  assert.equal(progress.rows[2].result,'wait');
});
test('mission completion requires disabled confirmation and no motion fault', () => {
  const p={running:true,phase:'complete',disabled_confirmed:false,stages:['post_action'],post:{phase:'complete'}};
  assert.notEqual(view.missionProgress(p,null).headline,'本轮完成');
  p.disabled_confirmed=true;
  const result=view.missionProgress(p,null);
  assert.equal(result.headline,'本轮完成');
  assert.equal(result.rows[7].result,'ok');
  assert.equal(view.missionProgress(p,{code:'STOP_UNCONFIRMED'}).rows[7].result,'fail');
});
test('read-only, paused, plan-only and expired states never claim active execution', () => {
  assert.equal(view.missionProgress({observe_only:true},null).headline,'只读观察');
  const p={running:true,phase:'post_left_translate_timed',stages:['post_action'],timing:{seconds:2}};
  assert.match(view.missionProgress(p,null).detail,/2 秒/);
  const stale=view.missionProgress(p,null,2.01);
  assert.equal(stale.headline,'状态待更新');
  assert.equal(stale.rows.some(r=>r.result==='run'),false);
  assert.equal(view.missionProgress({...p,phase:'stopped'},null).headline,'已暂停');
  const plan=view.missionProgress({phase:'awaiting_execution_confirmation',disabled_confirmed:true,profile:{execute_post:false}},null);
  assert.equal(plan.headline,'等待后续操作');
  for(const i of [2,4,5,6])assert.equal(plan.rows[i].result,'skip');
  assert.equal(plan.rows[7].result,'wait');
});
test('resumed post-actions do not invent acquisition or repeated lateral travel', () => {
 const p={phase:'complete',disabled_confirmed:true,post:{phase:'complete',initial_state:{slide_position_mm:150},
   lateral_completion:'skipped_C_already_extended_no_repeated_motion',B_top_confirmed:true}};
 const result=view.missionProgress(p,null);
 for(const i of [0,1,4]) assert.equal(result.rows[i].result,'skip');
});
test('unknown phase names cannot fabricate progress through prototype properties', () => {
 const p={running:true,phase:'constructor',stages:['constructor']};
 const result=view.missionProgress(p,null);
 assert.equal(result.rows.some(r=>r.result==='ok'),false);
 assert.equal(view.missionProgress({running:true,phase:'complete'},null).rows[7].detail,'等待禁用确认');
});
test('existing workers without a lifecycle journal still show their real tracking phase', () => {
 assert.equal(view.missionProgress({running:true,phase:'driving'},null).headline,'追踪并靠近');
});
