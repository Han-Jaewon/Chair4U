from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import logging

class AdvancedImageScraper:
    def __init__(self):
        self.driver = None
        self.setup_driver()
    
    def setup_driver(self):
        """Chrome 드라이버 설정"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # 브라우저 창 숨기기
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            logging.error(f"Chrome 드라이버 설정 실패: {e}")
            raise
    
    def get_image_url_selenium(self, product_url):
        """Selenium으로 이미지 URL 추출"""
        try:
            logging.info(f"Selenium으로 페이지 로드: {product_url}")
            self.driver.get(product_url)
            
            # 페이지 로딩 대기
            time.sleep(3)
            
            # 이미지 요소들 찾기
            img_selectors = [
                'img[src*="ohousecdn.com"]',
                'img[data-src*="ohousecdn.com"]',
                '.production-selling-header img',
                '.production-image img'
            ]
            
            for selector in img_selectors:
                try:
                    imgs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for img in imgs:
                        src = img.get_attribute('src') or img.get_attribute('data-src')
                        if src and 'ohousecdn.com' in src:
                            # 고해상도로 변환
                            if '?' in src:
                                base_url = src.split('?')[0]
                            else:
                                base_url = src
                            high_res = f"{base_url}?w=1280&h=1280&c=c&webp=1"
                            logging.info(f"Selenium으로 이미지 발견: {high_res}")
                            return high_res
                except Exception as e:
                    logging.warning(f"셀렉터 {selector} 실패: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logging.error(f"Selenium 스크래핑 실패: {e}")
            return None
    
    def close(self):
        """드라이버 종료"""
        if self.driver:
            self.driver.quit()

# 사용 예시
def test_selenium_method():
    scraper = None
    try:
        scraper = AdvancedImageScraper()
        test_url = "https://ohou.se/productions/444838/selling?affect_type=ProductCategoryIndex"
        result = scraper.get_image_url_selenium(test_url)
        print(f"결과: {result}")
    except Exception as e:
        print(f"오류: {e}")
    finally:
        if scraper:
            scraper.close()