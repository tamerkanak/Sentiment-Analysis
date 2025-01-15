# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import json

# Başlangıç bilgileri
BASE_URLS = [
    "https://denebunu.com/products/makyaj/?page={}",
    "https://denebunu.com/products/sac-bakim/?page={}",
    "https://denebunu.com/products/cilt-bakim/?page={}",
    "https://denebunu.com/products/yuz-bakim/?page={}"
]
MAX_PRODUCTS = 200
MAX_REVIEWS_PER_PRODUCT = 50  # Ürün başına 50 yorum, her rating için 10 yorum olacak
RATING_VALUES = [1, 2, 3, 4, 5]  # 1-5 arasında puanlar

def fetch_product_links(base_url, max_pages=10):
    """Ürün bağlantılarını çeker."""
    product_links = []
    print(f"[INFO] {base_url} sayfasından ürün bağlantıları toplanıyor...")
    for page in range(1, max_pages + 1):
        url = base_url.format(page)
        print(f"[DEBUG] Sayfa: {page} | URL: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[WARNING] Sayfa yüklenemedi: {url}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.select('div.thumbnail.mobileProducts > a')
        for link in links:
            href = link.get('href')
            if href:
                product_links.append(f"https://denebunu.com{href}")

        print(f"[INFO] {len(links)} ürün bulundu.")
        if len(product_links) >= MAX_PRODUCTS:
            print("[INFO] Yeterli ürün bağlantısı toplandı.")
            break

    return product_links[:MAX_PRODUCTS]

[
    {"text": "Bu ürünü çok beğendim.", "stars": 5},
    {"text": "Fena değildi.", "stars": 3}
]

def fetch_reviews_from_page(product_url, rating_value):
    """Bir ürünün belirli bir sayfasındaki yorumları ve yıldızları çeker."""
    reviews = []
    page_url = f"{product_url}?o=auto&r={rating_value}&sayfa=1"  # Rating ve 1. sayfa URL düzeni
    print(f"[INFO] Ürün yorumları çekiliyor: {page_url}")
    
    try:
        response = requests.get(page_url)
        if response.status_code != 200:
            print(f"[WARNING] Sayfa yüklenemedi: {page_url}")
            return reviews

        soup = BeautifulSoup(response.content, 'html.parser')
        review_elements = soup.select('div.panel.panel-review')  # Yorum elemanlarını seç
        for review in review_elements:
            # Yorum metni
            text_tag = review.select_one("div.panel-body")
            if text_tag:
                review_text = text_tag.get_text(strip=True)

                # Başlıkta bulunan istemediğimiz metinleri kaldır
                unwanted_texts = [
                    "Bu ürün için 5 üzerinden puanın nedir? (1: Çok Kötü, 5: Çok İyi)",
                    "Bu ürünü genel olarak nasıl değerlendirirsin?",
                    "Bu ürün Denebunu üzerinden Denebunu\n                                    kullanıcısı tarafından alınmış ve Denebunu kullanıcısı tarafından\n                                    değerlendirilmiştir."
                ]
                for unwanted_text in unwanted_texts:
                    review_text = review_text.replace(unwanted_text, "").strip()

                # Puan (yıldız sayısı)
                stars = len(review.select(".fa-star"))

                # Yorum ve puanı kaydet
                reviews.append({'text': review_text, 'stars': stars})

        print(f"[INFO] {rating_value} puan için {len(reviews)} yorum çekildi.")
    except Exception as e:
        print(f"[ERROR] Yorumlar çekilirken hata oluştu: {e}")

    return reviews

def fetch_reviews(product_url):
    """Bir ürüne ait tüm yorumları çeker."""
    all_reviews = []
    for rating in RATING_VALUES:
        print(f"[INFO] {rating} puan için yorumlar çekiliyor.")
        reviews = fetch_reviews_from_page(product_url, rating)
        if reviews:
            all_reviews.extend(reviews)
        else:
            print(f"[INFO] {product_url} için {rating} puanlı yorum bulunamadı.")
        
        if len(all_reviews) >= MAX_REVIEWS_PER_PRODUCT:
            print(f"[INFO] Maksimum yorum sayısına (50) ulaşıldı.")
            break

    return all_reviews

def main():
    """Ana işlev."""
    all_product_links = []
    # Üç sayfadan ürün bağlantılarını topla
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_product_links, base_url): base_url for base_url in BASE_URLS}

        for future in futures:
            product_links = future.result()
            all_product_links.extend(product_links)
            print(f"[INFO] {len(product_links)} ürün bağlantısı toplandı.")

    print(f"[INFO] Toplam {len(all_product_links)} ürün bağlantısı toplandı.")

    all_reviews = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_reviews, link): link for link in all_product_links}

        for future in futures:
            product_reviews = future.result()
            all_reviews.extend(product_reviews)
            print(f"[INFO] Şimdiye kadar toplam {len(all_reviews)} yorum toplandı.")

            if len(all_reviews) >= 40000:
                print("[INFO] Yeterli yorum toplandı. İşlem sonlandırılıyor.")
                break

    print(f"[INFO] Toplam {len(all_reviews)} yorum toplandı.")
    # Yorumları JSON dosyasına kaydet
    with open("review_balanced.json", "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, ensure_ascii=False, indent=4)
    print("[INFO] Yorumlar 'review_balanced.json' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
