// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HeroPage from './pages/HeroPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HeroPage />} />
        <Route path="/upload" element={<div>업로드 페이지 준비 중</div>} />
        <Route path="/select" element={<div>옵션 선택</div>} />
        <Route path="/result" element={<div>추천 결과</div>} />
      </Routes>
    </Router>
  );
}

export default App;