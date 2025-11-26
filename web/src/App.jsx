import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import EnrollPageV2 from './components/EnrollPageV2';
import VerifyPageV2 from './components/VerifyPageV2';
import SettingsPage from './components/SettingsPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<EnrollPageV2 />} />
            <Route path="/enroll" element={<EnrollPageV2 />} />
            <Route path="/verify" element={<VerifyPageV2 />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;