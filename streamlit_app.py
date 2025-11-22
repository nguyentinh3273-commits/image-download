import streamlit as st
import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
import base64
import json
import time
from typing import List, Dict, Any, Tuple, Optional

# --- Cấu hình API Gemini ---
# KHÔNG CẦN CHỈ ĐỊNH API KEY. Streamlit Cloud sẽ tự động cung cấp trong môi trường chạy.
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
API_KEY = "" # Sẽ được Canvas cung cấp tự động

# --- Thiết lập giao diện Streamlit ---
st.set_page_config(
    page_title="Trình Tải & Phân tích Hình ảnh Web",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #45a049;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

st.title("📸 Công cụ Tải và Phân tích Hình ảnh Web")
st.markdown("Dán URL của trang web bạn muốn trích xuất hình ảnh, sau đó sử dụng các bộ lọc và công cụ AI.")

# Khởi tạo state session
if 'extracted_images' not in st.session_state:
    st.session_state.extracted_images = []
if 'analyzed_images' not in st.session_state:
    st.session_state.analyzed_images = []


# --- Các Hàm Tiện ích ---

def base64_to_inline_data(image_base64: str, mime_type: str = "image/jpeg") -> Dict[str, Dict[str, str]]:
    """Tạo cấu trúc inlineData cho API Gemini."""
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": image_base64
        }
    }

def get_image_data_and_base64(img_url: str) -> Optional[Tuple[bytes, int, int, str]]:
    """
    Tải ảnh, lấy kích thước và chuyển đổi thành base64.
    Trả về (bytes, width, height, base64_string) hoặc None nếu thất bại.
    """
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        img_bytes = response.content
        img_mime = response.headers.get('Content-Type', 'image/jpeg')

        # Dùng PIL để lấy kích thước
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size

        # Chuyển đổi thành base64 cho API AI
        buffered = io.BytesIO()
        # Lưu lại dưới dạng JPEG để đảm bảo định dạng tương thích với API, nếu không phải GIF/PNG
        if 'image/png' in img_mime or 'image/gif' in img_mime:
             img.save(buffered, format=img.format or "PNG")
             img_mime = "image/png"
        else:
             img.save(buffered, format="JPEG")
             img_mime = "image/jpeg"
        
        base64_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_bytes, width, height, base64_encoded, img_mime

    except Exception as e:
        # print(f"Lỗi khi xử lý ảnh {img_url}: {e}")
        return None

# --- Chức năng Web Scraping ---

def extract_images(url: str):
    """Lấy tất cả các URL hình ảnh từ một trang web."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        st.info(f"Đang cố gắng tải nội dung từ: {url}")
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        img_tags = soup.find_all('img')
        
        # Lọc và chuẩn hóa URL
        image_urls = []
        for img in img_tags:
            src = img.get('src') or img.get('data-src')
            if src and src.startswith(('http', 'https')):
                image_urls.append(src)
            elif src and not src.startswith(('mailto', 'tel', '#')):
                # Xử lý URL tương đối
                from urllib.parse import urljoin
                full_url = urljoin(url, src)
                image_urls.append(full_url)
        
        # Loại bỏ các URL trùng lặp
        unique_urls = list(set(image_urls))
        st.success(f"Đã trích xuất được {len(unique_urls)} URL hình ảnh duy nhất.")
        
        return unique_urls

    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi truy cập URL: {e}")
        return []
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
        return []

# --- Chức năng Phân tích Hình ảnh (AI) ---

def analyze_image_with_ai(base64_data: str, mime_type: str, retry_count: int = 5) -> str:
    """Gọi API Gemini để phân tích hình ảnh và trả về mô tả."""
    
    prompt = "Mô tả chi tiết và chính xác hình ảnh này bằng tiếng Việt. Tập trung vào các đối tượng, hành động, và bối cảnh."
    
    # Chuẩn bị payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    base64_to_inline_data(base64_data, mime_type)
                ]
            }
        ],
    }

    headers = {'Content-Type': 'application/json'}
    
    for i in range(retry_count):
        try:
            # st.info(f"Thử gọi API lần {i + 1}...")
            response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status() # Ném ngoại lệ cho các mã lỗi HTTP 4xx/5xx
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                text = candidate['content']['parts'][0].get('text', 'Không thể tạo mô tả.')
                return text
            else:
                # Nếu không có text, thử lại hoặc trả về lỗi
                return "API trả về cấu trúc rỗng hoặc không hợp lệ."

        except requests.exceptions.RequestException as e:
            if response.status_code in [429, 500, 503] and i < retry_count - 1:
                wait_time = 2 ** i
                # st.warning(f"Lỗi tạm thời ({response.status_code}). Đợi {wait_time}s trước khi thử lại.")
                time.sleep(wait_time)
            else:
                # st.error(f"Lỗi gọi API sau {i + 1} lần thử: {e}")
                return f"Lỗi gọi API: {e}"
        except Exception as e:
            # st.error(f"Lỗi không xác định: {e}")
            return f"Lỗi không xác định: {e}"
            
    return "Thử lại thất bại. Vui lòng kiểm tra API key hoặc đợi một lát."


# --- Giao diện và Logic chính ---

with st.sidebar:
    st.header("⚙️ Thiết lập & Bộ lọc")
    input_url = st.text_input("URL Trang Web", "https://unsplash.com")

    st.subheader("Lọc Kích thước Hình ảnh (Pixels)")
    col1, col2 = st.columns(2)
    with col1:
        min_width = st.number_input("Chiều rộng tối thiểu (Min Width)", min_value=0, value=300)
    with col2:
        max_width = st.number_input("Chiều rộng tối đa (Max Width)", min_value=0, value=9999)
    
    col3, col4 = st.columns(2)
    with col3:
        min_height = st.number_input("Chiều cao tối thiểu (Min Height)", min_value=0, value=300)
    with col4:
        max_height = st.number_input("Chiều cao tối đa (Max Height)", min_value=0, value=9999)

    if st.button("Trích xuất Hình ảnh", use_container_width=True):
        if not input_url:
            st.error("Vui lòng nhập một URL hợp lệ.")
        else:
            with st.spinner("Đang trích xuất và kiểm tra kích thước hình ảnh..."):
                st.session_state.extracted_images = []
                st.session_state.analyzed_images = []
                
                urls = extract_images(input_url)
                
                # Bắt đầu tải và kiểm tra kích thước
                progress_bar = st.progress(0)
                image_data_list = []
                total_urls = len(urls)

                for i, img_url in enumerate(urls):
                    data = get_image_data_and_base64(img_url)
                    
                    if data:
                        img_bytes, width, height, base64_encoded, mime_type = data
                        
                        # Kiểm tra bộ lọc
                        if (min_width <= width <= max_width) and (min_height <= height <= max_height):
                            image_data_list.append({
                                "url": img_url,
                                "bytes": img_bytes,
                                "width": width,
                                "height": height,
                                "base64": base64_encoded,
                                "mime_type": mime_type,
                                "analysis": "Chưa phân tích"
                            })
                    
                    progress_bar.progress((i + 1) / total_urls)
                
                st.session_state.extracted_images = image_data_list
                st.success(f"Hoàn tất trích xuất. Đã tìm thấy và lọc được {len(image_data_list)} ảnh thỏa mãn bộ lọc.")
                progress_bar.empty()

# --- Hiển thị kết quả và Chức năng Phân tích AI ---

if st.session_state.extracted_images:
    
    # Hiển thị số lượng ảnh
    st.subheader(f"Kết quả Lọc: {len(st.session_state.extracted_images)} Hình ảnh")

    # Nút Phân tích AI
    if st.button("🤖 Phân tích Hình ảnh bằng AI", disabled=not st.session_state.extracted_images):
        st.session_state.analyzed_images = []
        with st.spinner("Đang gọi API Gemini để phân tích hình ảnh..."):
            
            analysis_progress = st.progress(0)
            
            for i, img_info in enumerate(st.session_state.extracted_images):
                
                description = analyze_image_with_ai(img_info['base64'], img_info['mime_type'])
                
                analyzed_item = img_info.copy()
                analyzed_item['analysis'] = description
                st.session_state.analyzed_images.append(analyzed_item)
                
                analysis_progress.progress((i + 1) / len(st.session_state.extracted_images))
            
            st.success("Hoàn tất phân tích AI cho tất cả hình ảnh đã lọc.")
            analysis_progress.empty()

    
    # --- Hiển thị ảnh và Metadata ---
    
    display_list = st.session_state.analyzed_images if st.session_state.analyzed_images else st.session_state.extracted_images
    
    st.markdown("---")
    st.subheader("Xem trước và Tải hàng loạt")
    
    # Tải ảnh hàng loạt (tạo tệp ZIP ảo)
    if display_list:
        from io import BytesIO
        import zipfile
        
        @st.cache_data
        def create_zip_archive(data_list):
            """Tạo tệp ZIP trong bộ nhớ."""
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for i, item in enumerate(data_list):
                    # Đặt tên file bằng index + kích thước + mô tả (nếu có)
                    file_name_prefix = f"{i+1}_{item['width']}x{item['height']}"
                    
                    # Nếu có mô tả AI, thêm nó vào một tệp TXT
                    if 'analysis' in item and item['analysis'] != 'Chưa phân tích':
                         txt_content = f"URL gốc: {item['url']}\nKích thước: {item['width']}x{item['height']} pixels\nMô tả AI:\n{item['analysis']}"
                         zip_file.writestr(f"{file_name_prefix}_description.txt", txt_content.encode('utf-8'))

                    # Thêm hình ảnh
                    ext = ".jpg" if "jpeg" in item['mime_type'] else ".png" if "png" in item['mime_type'] else ".bin"
                    zip_file.writestr(f"{file_name_prefix}{ext}", item['bytes'])
            
            return zip_buffer.getvalue()

        zip_bytes = create_zip_archive(display_list)
        
        st.download_button(
            label=f"⬇️ Tải {len(display_list)} Ảnh và Mô tả (ZIP)",
            data=zip_bytes,
            file_name="trich_xuat_hinh_anh.zip",
            mime="application/zip",
            use_container_width=False
        )
        st.markdown(f"*(Tệp ZIP chứa {len(display_list)} ảnh và {len(st.session_state.analyzed_images)} tệp mô tả AI nếu đã phân tích)*")


    # Hiển thị từng ảnh trong một bố cục lưới
    cols = st.columns(3)
    
    for i, item in enumerate(display_list):
        col = cols[i % 3]
        
        with col:
            # Tạo đường dẫn data URL để hiển thị
            img_data_url = f"data:{item['mime_type']};base64,{item['base64']}"
            st.image(img_data_url, caption=f"Kích thước: {item['width']}x{item['height']}", use_column_width=True)
            
            if item.get('analysis', 'Chưa phân tích') != 'Chưa phân tích':
                with st.expander("Phân tích AI"):
                    st.markdown(item['analysis'])
            
            # Tải từng ảnh riêng lẻ
            st.download_button(
                label="Tải xuống",
                data=item['bytes'],
                file_name=f"image_{i+1}_{item['width']}x{item['height']}.jpg",
                mime=item['mime_type'],
                key=f"download_{i}",
                use_container_width=True
            )
            st.markdown("---")

else:
    if 'extracted_images' in st.session_state and st.session_state.extracted_images == [] and input_url:
         st.warning("Không tìm thấy hình ảnh nào thỏa mãn bộ lọc của bạn trên trang web này.")
