const BASE = '/lotto/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); } catch { data = null; }
  if (!res.ok) throw new Error(data?.error || `요청 실패 (${res.status})`);
  if (!data) throw new Error('서버 응답이 비어있습니다.');
  return data;
}

export function login(userId, password) {
  return request('/login', {
    method: 'POST',
    body: JSON.stringify({ userId, password }),
  });
}

export function getBalance(userId, password) {
  return request('/balance', {
    method: 'POST',
    body: JSON.stringify({ userId, password }),
  });
}

export function purchase(userId, password) {
  return request('/purchase', {
    method: 'POST',
    body: JSON.stringify({ userId, password }),
  });
}

export function getHistory(userId) {
  return request(`/history?userId=${encodeURIComponent(userId)}`);
}

export function getLatest() {
  return request('/latest');
}

export function getRestriction() {
  return request('/restriction');
}

export function getPrediction() {
  return request('/prediction');
}

export function refreshLatest() {
  return request('/latest/refresh', { method: 'POST' });
}

export function refreshPrediction() {
  return request('/prediction/refresh', { method: 'POST' });
}

export function getModelStatus() {
  return request('/model/status');
}

export function trainModel() {
  return request('/model/train', { method: 'POST' });
}
