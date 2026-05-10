import { useState, useEffect, useCallback } from 'react';
import {
  IconWallet, IconRefresh, IconCart, IconHistory,
  IconTrophy, IconLogout, IconWarning, IconClover, IconCheck, Spinner,
} from './Icons';
import { getBalance, purchase, getHistory, getLatest, getRestriction, getPrediction } from './api';
import './Dashboard.css';

// ─── Lotto Ball ───
function LottoBall({ number, size = 36, delay = 0 }) {
  const getColor = (n) => {
    if (n <= 10) return '#fbbf24';
    if (n <= 20) return '#3b82f6';
    if (n <= 30) return '#ef4444';
    if (n <= 40) return '#6b7280';
    return '#22c55e';
  };
  return (
    <div
      className="lotto-ball"
      style={{
        '--ball-color': getColor(number),
        width: size,
        height: size,
        fontSize: size * 0.4,
        animationDelay: `${delay}ms`,
      }}
    >
      {number}
    </div>
  );
}

// ─── Balance Card ───
function BalanceCard({ balance, onRefresh, refreshing }) {
  return (
    <div className="card balance-card animate-slide-up">
      <div className="card-header">
        <div className="card-title">
          <IconWallet size={20} color="var(--accent)" />
          <span>내 잔액</span>
        </div>
        <button className="icon-btn" onClick={onRefresh} disabled={refreshing} title="새로고침">
          <IconRefresh size={18} className={refreshing ? 'spinning' : ''} />
        </button>
      </div>
      <div className="balance-amounts">
        <div className="balance-row">
          <span className="balance-label">예치금</span>
          <span className="balance-value">{(balance?.deposit_balance ?? 0).toLocaleString()}원</span>
        </div>
        <div className="balance-row primary">
          <span className="balance-label">구매 가능</span>
          <span className="balance-value accent">{(balance?.available_amount ?? 0).toLocaleString()}원</span>
        </div>
      </div>
    </div>
  );
}

// ─── Restriction Banner ───
function RestrictionBanner({ restriction }) {
  if (!restriction || !restriction.restricted) return null;
  return (
    <div className="restriction-banner animate-fade-in">
      <IconWarning size={18} color="var(--warning)" />
      <span>{restriction.reason}</span>
    </div>
  );
}

// ─── Purchase Section ───
function PurchaseSection({ userId, password, restriction, onPurchased }) {
  const [state, setState] = useState('idle'); // idle | predicting | purchasing | done | error
  const [result, setResult] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    getPrediction().then(data => setPrediction(data)).catch(() => {});
  }, []);

  const handlePurchase = async () => {
    setState('predicting');
    setError('');
    try {
      await new Promise(r => setTimeout(r, 800));
      setState('purchasing');
      const res = await purchase(userId, password);
      setResult(res);
      setState('done');
      onPurchased?.();
    } catch (err) {
      setError(err.message);
      setState('error');
    }
  };

  const isDisabled = restriction?.restricted;

  return (
    <div className="card purchase-card animate-slide-up" style={{ animationDelay: '100ms' }}>
      <div className="card-header">
        <div className="card-title">
          <IconCart size={20} color="var(--accent)" />
          <span>AI 자동 구매</span>
        </div>
      </div>

      {prediction?.games && state === 'idle' && (
        <div className="prediction-preview">
          <p className="prediction-label">{prediction.games.target_round}회차 예측 번호</p>
          <div className="prediction-games">
            {prediction.games.games.map((game, i) => (
              <div key={i} className="game-row">
                <span className="game-label">G{i + 1}</span>
                <div className="game-balls">
                  {game.map((n, j) => <LottoBall key={j} number={n} size={30} delay={j * 60} />)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {state === 'idle' && (
        <button className="purchase-btn" onClick={handlePurchase} disabled={isDisabled}>
          <IconCart size={20} />
          {isDisabled ? '구매 불가' : '구매하기'}
        </button>
      )}

      {(state === 'predicting' || state === 'purchasing') && (
        <div className="purchase-progress">
          <div className="progress-steps">
            <div className={`progress-step ${state === 'predicting' ? 'active' : 'done'}`}>
              <div className="step-icon">
                {state === 'predicting' ? <Spinner size={20} /> : <IconCheck size={16} color="var(--success)" />}
              </div>
              <span>예측 번호 생성 중...</span>
            </div>
            <div className={`progress-step ${state === 'purchasing' ? 'active' : ''}`}>
              <div className="step-icon">
                {state === 'purchasing' ? <Spinner size={20} /> : <div className="step-dot" />}
              </div>
              <span>구매 진행 중...</span>
            </div>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: state === 'predicting' ? '40%' : '85%' }} />
          </div>
        </div>
      )}

      {state === 'done' && result && (
        <div className="purchase-result success animate-fade-in">
          <IconCheck size={24} color="var(--success)" />
          <p>구매 완료!</p>
          <div className="result-games">
            {result.games?.map((game, i) => (
              <div key={i} className="game-row">
                <span className="game-label">G{i + 1}</span>
                <div className="game-balls">
                  {game.map((n, j) => <LottoBall key={j} number={n} size={28} delay={j * 80} />)}
                </div>
              </div>
            ))}
          </div>
          <button className="btn-secondary" onClick={() => setState('idle')}>확인</button>
        </div>
      )}

      {state === 'error' && (
        <div className="purchase-result error animate-fade-in">
          <IconWarning size={24} color="var(--error)" />
          <p>{error}</p>
          <button className="btn-secondary" onClick={() => setState('idle')}>다시 시도</button>
        </div>
      )}
    </div>
  );
}

// ─── History Section ───
function HistorySection({ userId }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    getHistory(userId).then(setHistory).catch(() => {}).finally(() => setLoading(false));
  }, [userId]);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="card animate-slide-up" style={{ animationDelay: '200ms' }}>
      <div className="card-header">
        <div className="card-title">
          <IconHistory size={20} color="var(--accent)" />
          <span>구매 내역</span>
        </div>
        <button className="icon-btn" onClick={load} title="새로고침">
          <IconRefresh size={18} className={loading ? 'spinning' : ''} />
        </button>
      </div>

      {loading && <div className="skeleton" style={{ height: 60 }} />}

      {!loading && history.length === 0 && (
        <div className="empty-state">
          <IconHistory size={32} color="var(--text-muted)" />
          <p>구매 내역이 없습니다</p>
        </div>
      )}

      {!loading && history.map((item, i) => (
        <div key={i} className="history-item">
          <div className="history-meta">
            <span className={`history-status ${item.success ? 'success' : 'fail'}`}>
              {item.success ? '성공' : '실패'}
            </span>
            <span className="history-date">
              {new Date(item.timestamp).toLocaleString('ko-KR', { timeZone: 'Asia/Seoul' })}
            </span>
          </div>
          {item.games?.map((game, gi) => (
            <div key={gi} className="game-row compact">
              <div className="game-balls">
                {game.map((n, j) => <LottoBall key={j} number={n} size={26} />)}
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// ─── Latest Results ───
function LatestResults() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getLatest().then(setResults).catch(() => {}).finally(() => setLoading(false));
  }, []);

  return (
    <div className="card animate-slide-up" style={{ animationDelay: '300ms' }}>
      <div className="card-header">
        <div className="card-title">
          <IconTrophy size={20} color="var(--accent)" />
          <span>최근 당첨번호</span>
        </div>
      </div>

      {loading && [1, 2, 3].map(i => <div key={i} className="skeleton" style={{ height: 48, marginBottom: 8 }} />)}

      {!loading && results.map((r, i) => (
        <div key={i} className="result-row" style={{ animationDelay: `${i * 60}ms` }}>
          <div className="result-meta">
            <span className="result-round">{r['회차']}회</span>
            <span className="result-date">{r['추첨일']}</span>
          </div>
          <div className="game-balls">
            {r['당첨번호'].map((n, j) => <LottoBall key={j} number={n} size={30} delay={j * 50} />)}
            <span className="bonus-sep">+</span>
            <LottoBall number={r['보너스번호']} size={30} delay={350} />
          </div>
          <div className="result-prize">{r['1등 당첨금']}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Main Dashboard ───
export default function Dashboard({ userId, password, initialBalance, onLogout }) {
  const [balance, setBalance] = useState(initialBalance);
  const [refreshing, setRefreshing] = useState(false);
  const [restriction, setRestriction] = useState(null);
  const [tab, setTab] = useState('home');

  useEffect(() => {
    getRestriction().then(setRestriction).catch(() => {});
  }, []);

  const refreshBalance = useCallback(async () => {
    setRefreshing(true);
    try {
      const data = await getBalance(userId, password);
      setBalance(data);
    } catch {}
    setRefreshing(false);
  }, [userId, password]);

  const handleLogout = () => {
    sessionStorage.removeItem('lotto_session');
    onLogout();
  };

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dash-header">
        <div className="header-left">
          <IconClover size={24} color="var(--accent)" />
          <h1>LOTTO AI</h1>
        </div>
        <button className="icon-btn logout-btn" onClick={handleLogout} title="로그아웃">
          <IconLogout size={20} />
        </button>
      </header>

      {/* Content */}
      <main className="dash-content">
        <RestrictionBanner restriction={restriction} />

        {tab === 'home' && (
          <>
            <BalanceCard balance={balance} onRefresh={refreshBalance} refreshing={refreshing} />
            <PurchaseSection
              userId={userId}
              password={password}
              restriction={restriction}
              onPurchased={refreshBalance}
            />
            <LatestResults />
          </>
        )}

        {tab === 'history' && <HistorySection userId={userId} />}
      </main>

      {/* Bottom Nav */}
      <nav className="bottom-nav">
        <button className={`nav-item ${tab === 'home' ? 'active' : ''}`} onClick={() => setTab('home')}>
          <IconClover size={22} />
          <span>홈</span>
        </button>
        <button className={`nav-item ${tab === 'history' ? 'active' : ''}`} onClick={() => setTab('history')}>
          <IconHistory size={22} />
          <span>내역</span>
        </button>
      </nav>
    </div>
  );
}
