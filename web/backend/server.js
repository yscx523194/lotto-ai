import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const __dirname = dirname(fileURLToPath(import.meta.url));
const LOTTO_ROOT = join(__dirname, '..', '..');
const LOTTO_SRC = join(LOTTO_ROOT, 'lotto', 'src');
const DATA_DIR = join(__dirname, 'data');
const HISTORY_FILE = join(DATA_DIR, 'purchase_history.json');
const CACHE_FILE = join(LOTTO_ROOT, 'lotto_cache.json');

// Ensure data dir exists
import { mkdirSync } from 'fs';
if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });
if (!existsSync(HISTORY_FILE)) writeFileSync(HISTORY_FILE, '[]', 'utf-8');

const app = express();
app.use(cors());
app.use(helmet({ contentSecurityPolicy: false }));
app.use(express.json());

// ─── Helpers ───
function readHistory() {
  try {
    return JSON.parse(readFileSync(HISTORY_FILE, 'utf-8'));
  } catch { return []; }
}

function writeHistory(data) {
  writeFileSync(HISTORY_FILE, JSON.stringify(data, null, 2), 'utf-8');
}

function runPython(script, args = []) {
  return execFileAsync('python3', [script, ...args], {
    cwd: LOTTO_SRC,
    timeout: 120_000,
    env: { ...process.env, HEADLESS: 'true' },
  });
}

function getPurchaseRestriction() {
  const now = new Date();
  const kst = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
  const day = kst.getDay(); // 0=Sun, 6=Sat
  const hour = kst.getHours();

  if (day === 0 || day === 6) {
    return { restricted: true, reason: '토요일/일요일에는 자동구매가 불가능합니다.' };
  }
  if (hour >= 20) {
    return { restricted: true, reason: '오후 8시 이후에는 구매가 불가능합니다.' };
  }
  return { restricted: false };
}

// ─── API Routes ───

// Login: verify credentials by running balance check
app.post('/api/login', async (req, res) => {
  const { userId, password } = req.body;
  if (!userId || !password) {
    return res.status(400).json({ error: '아이디와 비밀번호를 입력하세요.' });
  }

  try {
    // Write temp .env for this login
    const envPath = join(LOTTO_ROOT, 'lotto', '.env');
    writeFileSync(envPath, `USER_ID=${userId}\nPASSWD=${password}\n`);

    const { stdout } = await runPython('balance.py');
    const match = stdout.match(/__RESULT__\s*({.*})/);
    if (match) {
      const result = JSON.parse(match[1]);
      if (result.status === 'SUCCESS') {
        return res.json({
          success: true,
          balance: result.detail,
        });
      }
    }
    return res.status(401).json({ error: '로그인에 실패했습니다.' });
  } catch (err) {
    console.error('Login error:', err.message);
    return res.status(500).json({ error: '로그인 처리 중 오류가 발생했습니다.' });
  }
});

// Balance check
app.post('/api/balance', async (req, res) => {
  const { userId, password } = req.body;
  if (!userId || !password) {
    return res.status(400).json({ error: '인증 정보가 필요합니다.' });
  }

  try {
    const envPath = join(LOTTO_ROOT, 'lotto', '.env');
    writeFileSync(envPath, `USER_ID=${userId}\nPASSWD=${password}\n`);

    const { stdout } = await runPython('balance.py');
    const match = stdout.match(/__RESULT__\s*({.*})/);
    if (match) {
      const result = JSON.parse(match[1]);
      if (result.status === 'SUCCESS') {
        return res.json(result.detail);
      }
    }
    return res.status(500).json({ error: '잔액 조회에 실패했습니다.' });
  } catch (err) {
    console.error('Balance error:', err.message);
    return res.status(500).json({ error: '잔액 조회 중 오류가 발생했습니다.' });
  }
});

// Purchase
app.post('/api/purchase', async (req, res) => {
  const { userId, password } = req.body;
  if (!userId || !password) {
    return res.status(400).json({ error: '인증 정보가 필요합니다.' });
  }

  // Time restriction check
  const restriction = getPurchaseRestriction();
  if (restriction.restricted) {
    return res.status(403).json({ error: restriction.reason });
  }

  try {
    const envPath = join(LOTTO_ROOT, 'lotto', '.env');
    writeFileSync(envPath, `USER_ID=${userId}\nPASSWD=${password}\n`);

    // 1. Generate predictions
    const purchaseGamesPath = join(LOTTO_ROOT, 'models_v2', 'purchase_games.json');
    let games;
    if (existsSync(purchaseGamesPath)) {
      const saved = JSON.parse(readFileSync(purchaseGamesPath, 'utf-8'));
      games = saved.games;
    } else {
      return res.status(500).json({ error: '예측 모델이 준비되지 않았습니다. git pull을 실행하세요.' });
    }

    // 2. Run purchase via lotto645.py
    const { stdout } = await runPython('lotto645.py', [JSON.stringify(games)]);
    const match = stdout.match(/__RESULT__\s*({.*})/);

    const history = readHistory();
    const record = {
      userId,
      timestamp: new Date().toISOString(),
      games,
      success: false,
      detail: null,
    };

    if (match) {
      const result = JSON.parse(match[1]);
      record.success = result.status === 'SUCCESS';
      record.detail = result.detail || result;
    }

    history.push(record);
    writeHistory(history);

    if (record.success) {
      return res.json({ success: true, games, detail: record.detail });
    }
    return res.status(500).json({ error: '구매 처리 중 오류가 발생했습니다.', games });
  } catch (err) {
    console.error('Purchase error:', err.message);
    return res.status(500).json({ error: '구매 처리 중 오류가 발생했습니다.' });
  }
});

// Purchase history
app.get('/api/history', (req, res) => {
  const userId = req.query.userId;
  const history = readHistory();
  const filtered = userId
    ? history.filter(h => h.userId === userId)
    : history;
  res.json(filtered.reverse());
});

// Latest winning numbers
app.get('/api/latest', (_req, res) => {
  try {
    if (!existsSync(CACHE_FILE)) {
      return res.status(404).json({ error: '당첨번호 데이터가 없습니다.' });
    }
    const cache = JSON.parse(readFileSync(CACHE_FILE, 'utf-8'));
    const entries = Object.values(cache).sort((a, b) => b['회차'] - a['회차']);
    const latest = entries.slice(0, 10);
    res.json(latest);
  } catch (err) {
    console.error('Latest error:', err.message);
    res.status(500).json({ error: '당첨번호 조회 중 오류가 발생했습니다.' });
  }
});

// Purchase restriction info
app.get('/api/restriction', (_req, res) => {
  res.json(getPurchaseRestriction());
});

// Prediction info
app.get('/api/prediction', (_req, res) => {
  try {
    const predPath = join(LOTTO_ROOT, 'models_v2', 'prediction_v2.json');
    const gamesPath = join(LOTTO_ROOT, 'models_v2', 'purchase_games.json');
    const result = {};
    if (existsSync(gamesPath)) {
      result.games = JSON.parse(readFileSync(gamesPath, 'utf-8'));
    }
    if (existsSync(predPath)) {
      result.prediction = JSON.parse(readFileSync(predPath, 'utf-8'));
    }
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: '예측 데이터 조회 실패' });
  }
});

const PORT = process.env.PORT || 3003;
app.listen(PORT, () => {
  console.log(`Lotto API server running on port ${PORT}`);
});
