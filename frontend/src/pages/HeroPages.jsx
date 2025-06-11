import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function HeroPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-white">
      {/* GNB */}
      <header className="flex justify-between items-center px-8 py-4 border-b">
        <div
          className="text-2xl font-bold text-gray-800 cursor-pointer"
          onClick={() => navigate('/')}
        >
          Chair4U
        </div>
        <nav className="flex space-x-6 text-gray-700 text-sm">
          <button className="hover:text-blue-500">측정</button>
          <button className="hover:text-blue-500">둘러보기</button>
        </nav>
      </header>

      {/* Hero Section */}
      <main className="w-full min-h-[calc(100vh-80px)] flex items-center justify-center">
        <div className="w-full flex flex-col lg:flex-row items-center justify-between px-20 gap-12">
          {/* 텍스트 영역 */}
          <div className="flex-1 flex flex-col items-start justify-center max-w-2xl pl-16">
            <h1 className="text-7xl font-bold text-gray-800 mb-8">Chair4U</h1>
            <p className="text-3xl font-semibold text-gray-700 mb-6">
              당신만을 위한 의자 추천 시스템
            </p>
            <p className="text-xl text-gray-600 mb-10 leading-relaxed">
              더 이상 알아보지 않아도 됩니다.<br />
              AI가 당신의 체형을 분석합니다.<br />
              <span className="text-blue-600 font-semibold">사진 1장</span>으로 가능한 나만의 의자 피팅 서비스
            </p>
            <button
              className="bg-blue-500 text-white text-xl px-10 py-4 rounded-lg shadow-lg hover:bg-blue-600 transition-all duration-300 transform hover:scale-105"
              onClick={() => navigate('/upload')}
            >
              시작하기
            </button>
          </div>

          {/* 이미지 영역 - 오른쪽에 크게 */}
          <div className="relative w-[750px] h-[500px] -mr-20">
            {/* 사람 실루엣 - 뒤에 작게 배치 */}
            <img
              src="/assets/chair-intro2.png"
              alt="사람 측정 포인트"
              className="absolute top-0 left-[80px] w-[300px] h-auto z-10"
            />
            
            {/* 의자 이미지 - 화면 오른쪽 끝까지 */}
            <img
              src="/assets/chair-intro1.png"
              alt="인체공학 의자"
              className="absolute bottom-0 right-0 w-[480px] h-auto z-20"
            />
          </div>
        </div>
      </main>
    </div>
  );
}