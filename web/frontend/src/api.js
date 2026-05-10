const BASE = '/lotto/api';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || '요청 실패');
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
