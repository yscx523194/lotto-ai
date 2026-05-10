import { useState, useCallback } from 'react';
import Login from './Login';
import Dashboard from './Dashboard';

export default function App() {
  const [auth, setAuth] = useState(null);

  const handleLogin = useCallback((userId, password, balance) => {
    setAuth({ userId, password, balance });
  }, []);

  const handleLogout = useCallback(() => {
    setAuth(null);
  }, []);

  if (!auth) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <Dashboard
      userId={auth.userId}
      password={auth.password}
      initialBalance={auth.balance}
      onLogout={handleLogout}
    />
  );
}
