import { useState, useEffect, useCallback } from 'react';
import { IconUser, IconLock, IconClover } from './Icons';
import { login as apiLogin } from './api';
import './Login.css';

export default function Login({ onLogin }) {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [saveId, setSaveId] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const saved = localStorage.getItem('lotto_saved_id');
    if (saved) {
      setUserId(saved);
      setSaveId(true);
    }
    // Auto-login check
    const session = sessionStorage.getItem('lotto_session');
    if (session) {
      try {
        const data = JSON.parse(session);
        if (data.userId && data.password) {
          onLogin(data.userId, data.password, data.balance);
        }
      } catch {}
    }
  }, [onLogin]);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!userId.trim() || !password.trim()) {
      setError('아이디와 비밀번호를 입력하세요.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const result = await apiLogin(userId, password);
      if (saveId) {
        localStorage.setItem('lotto_saved_id', userId);
      } else {
        localStorage.removeItem('lotto_saved_id');
      }
      sessionStorage.setItem('lotto_session', JSON.stringify({
        userId, password, balance: result.balance,
      }));
      onLogin(userId, password, result.balance);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId, password, saveId, onLogin]);

  return (
    <div className="login-container">
      <div className="login-bg-orbs">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      <form className="login-card animate-fade-in" onSubmit={handleSubmit}>
        <div className="login-logo">
          <div className="logo-circle">
            <IconClover size={36} color="var(--accent)" />
          </div>
          <h1>LOTTO AI</h1>
          <p>AI 기반 로또 예측 시스템</p>
        </div>

        {error && (
          <div className="login-error animate-fade-in">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            {error}
          </div>
        )}

        <div className="input-group">
          <div className="input-icon"><IconUser size={18} color="var(--text-muted)" /></div>
          <input
            type="text"
            placeholder="아이디"
            value={userId}
            onChange={e => setUserId(e.target.value)}
            autoComplete="username"
            disabled={loading}
          />
        </div>

        <div className="input-group">
          <div className="input-icon"><IconLock size={18} color="var(--text-muted)" /></div>
          <input
            type="password"
            placeholder="비밀번호"
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoComplete="current-password"
            disabled={loading}
          />
        </div>

        <label className="save-id-label">
          <input
            type="checkbox"
            checked={saveId}
            onChange={e => setSaveId(e.target.checked)}
          />
          <span className="checkbox-custom" />
          <span>아이디 저장</span>
        </label>

        <button type="submit" className="login-btn" disabled={loading}>
          {loading ? (
            <span className="btn-loading">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style={{ animation: 'spin 1s linear infinite' }}>
                <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="3" />
                <path d="M12 2a10 10 0 0 1 10 10" stroke="white" strokeWidth="3" strokeLinecap="round" />
              </svg>
              로그인 중...
            </span>
          ) : '로그인'}
        </button>
      </form>
    </div>
  );
}
