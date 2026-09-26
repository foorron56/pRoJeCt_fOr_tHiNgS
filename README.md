# LifeStream AI

> Mắt điện bảo vệ bể bơi và vùng lũ — phát hiện đuối nước & người bị lũ cuốn
> theo thời gian thực bằng Computer Vision. / Real-time drowning & flood-sweep
> detection with Computer Vision.

[![tests](https://img.shields.io/badge/tests-52%20passing-brightgreen)]() 
![python](https://img.shields.io/badge/python-3.10%2B-blue)

## Vấn đề / Problem

Đuối nước là một trong những nguyên nhân tử vong hàng đầu ở trẻ em. Người cứu hộ
không thể nhìn mọi góc của bể bơi mọi khoảnh khắc — và nạn nhân đuối nước thường
**không kêu được, không vẫy tay được** (Phản ứng đuối nước bản năng — Instinctive
Drowning Response): họ đứng thẳng trong nước, điêu luyện lặng lẽ, rồi chìm.

**LifeStream AI** theo dõi hành vi mỗi người bơi qua camera thường, chấm điểm
nguy cơ đuối nước theo thời gian thực và phát cảnh báo cho cứu hộ viên trong
vài giây — trước khi nạn nhân chìm hẳn.

## Kiến trúc / Architecture

```
Camera / RTSP / video ─┐
                       ├─> SwimDetector (MediaPipe Pose + motion proposals)
    PoolSimulator ─────┘         │  SwimObservation (pose keypoints/bbox)
                                 v
                          CentroidTracker  (giữ định danh người bơi)
                                 v
                     SwimFeatureExtractor (cửa sổ ~1 s / người bơi)
                       vertical_posture | limb_distress | stillness
                       submersion       | head_dip     | isolation
                                 v
                      DrowningRiskEngine  (trọng số minh bạch)
                       score ∈ [0,1] + pha: SWIMMING → LAPS →
                       DISTRESS → STILL_INACTIVE → SINKING
                                 v
                 AlertManager (audit log + webhook Slack/Discord)
                                 v
                    HUD overlay (khung cảnh báo + độ trễ)
```

### Tín hiệu hành vi / Behavioral signals

| Tín hiệu | Ý nghĩa | Bằng chứng đuối nước |
|---|---|---|
| `vertical_posture` | Trục thân người bơi | Nạn nhân **đứng thẳng**, không bơi ngang được |
| `limb_distress`   | Biên độ vẫy tay/chân | Đập mạnh, cuống cuồng |
| `stillness`       | Chuyển động hiện tại vs. đỉnh gần nhất | Bỗng **im lặng** sau khi đang hoạt động |
| `submersion`      | Đầu ngập dưới mặt nước + drift xuống liên tục | Chìm dần |
| `head_dip`        | Đầu nhấp nhô chìm/nổi | Tách nạn nhân khỏi người *đứng nước* bình thường |
| `isolated`        | Khoảng cách người bơi gần nhất | Không ai gần để ứng cứu |

Điểm rủi ro là **tổng trọng số minh bạch** (tunable trong `lifestream/config.py`)
— mỗi điểm số đều giải thích được cho cứu hộ viên, không phải hộp đen.

## Chạy thử / Quickstart

```bash
pip install -r requirements.txt

# Windows: menu tương tác — chọn 1 (lũ) hoặc 2 (bể) sẽ tự phát MỘT VIDEO
# NGẪU NHIÊN trong thư mục data\; chọn 6 để tự tay chọn video từ danh sách
# data\ (run.bat flood <video> để chỉ định video trực tiếp)
run.bat

# === Hồ bơi / Pool ===
# Demo mô phỏng (không cần camera): người bơi đuối nước + người bơi vòng bộ
python -m lifestream --simulate --scenario distress --seconds 30

# Kịch bản kinh điển: đứng nước 6 giây → im lặng → chìm dần (phản ứng IDR)
python -m lifestream --simulate --scenario still --seconds 30

# Camera thật / RTSP / file video
python -m lifestream --source 0
python -m lifestream --source rtsp://pool-cam.local/stream
python -m lifestream --source demo_pool.mp4 --record demo/annotated.mp4

# === Lũ cuốn / Flood Guard ===
# Bị cuốn trôi: trượt chân ở bờ → bị nước kéo đi → kẹt trụ cầu
python -m lifestream --mode flood --simulate --flood-scenario drift

# Bám trụ giữa dòng, đầu chìm nhô liên tục; hoặc kẹt đá giữa dòng
python -m lifestream --mode flood --simulate --flood-scenario clinging
python -m lifestream --mode flood --simulate --flood-scenario stranded

# Camera thật / RTSP / video cho vùng lũ
python -m lifestream --mode flood --source 0
python -m lifestream --mode flood --source flood_footage.mp4 --record demo/out.mp4

# Webhook cảnh báo (Slack/Discord)
set LIFESTREAM_WEBHOOK=https://hooks.slack.com/services/...   (Windows)
python -m lifestream --simulate --scenario distress
```

Phím `q` để thoát cửa sổ xem trước.

### Kịch bản mô phỏng / Simulated scenarios

**Hồ bơi (pool):**

| Kịch bản | Nội dung | Kết quả mong đợi |
|---|---|---|
| `distress` | Đứng thẳng, vẫy cuống, đầu nhấn dưới nước | Cảnh báo ~3–5 s |
| `still`    | Đứng nước 6 s → im lặng → chìm dần 12 s | Cảnh báo khi chìm |
| `laps`     | Bơi tự do ngang qua bể | Không bao giờ cảnh báo |
| `mixed`    | distress 18 s → chuyển chìm | Cảnh báo duy trì |
| `playing`  | Nô đùa an toàn ở đầu bể nông | Không cảnh báo |

Kịch bản mặc định là `distress` (đổi bằng `--scenario`).

**Lũ cuốn (flood):**

| Kịch bản | Nội dung | Kết quả mong đợi |
|---|---|---|
| `drift`    | Trượt chân ở bờ → bị cuốn → kẹt trụ cầu, chìm nhô | Cảnh báo < 10 s |
| `clinging` | Bám trụ cầu, đầu chìm nhô mỗi ~3 s | Cảnh báo < 6 s |
| `stranded` | Kẹt trên đá giữa dòng, dòng 2.6 m/s cuốn qua | Cảnh báo pha STRANDED |
| `wading`   | Băng nước nông rồi đi bộ trên đường ngập | Không bao giờ cảnh báo |

## Kiểm thử / Tests

```bash
python -m pytest          # 52 tests: tracker, features, engine, alerts, e2e
                          # + Flood Guard + learned scorer & training pipeline
                          # + bộ chọn video (ngẫu nhiên / danh sách) cho run.bat
python scripts/train.py   # huấn luyện lại 2 model (models/*_scorer.json)
python scripts/verify.py  # in bảng pha/điểm theo thời gian mỗi kịch bản bể
```

Độ trễ xử lý trên CPU laptop: **< 5 ms/khung** cho pipeline bể; pipeline flood
gồm optical flow + render chạy < 120 ms/khung (8+ fps realtime).

## Machine Learning 🤖

Ngoài rule engine minh bạch, LifeStream có **learned scorer** đã huấn luyện:

```bash
python scripts/train.py                        # train cả hai model
python -m lifestream --scorer learned ...      # chạy với model đã học
```

- Dataset: hàng trăm nghìn frame từ các episode mô phỏng được **ngẫu nhiên hóa**
  (posture, tần số tay/chân, độ chìm đầu, dòng chảy, nhấn nước) — đi qua
  **đúng geometry simulator và feature extractor của runtime**, nhãn lấy từ
  ground-truth tham số vật lý chứ không phải từ feature đo được.
- Model: logistic regression chuẩn hóa, class-balanced, L2 — **numpy thuần**,
  huấn luyện vài giây, không thêm dependency. Mỗi điểm score vẫn giải thích
  được (w·x + b, contributions hiển thị trong HUD).
- Kết quả kiểm định (held-out episodes): pool F1 ≈ 0.97, flood F1 ≈ 0.78
  (engine vẫn giữ persistence gates nên fp 0 ở mọi kịch bản demo).
- Chi tiết: `docs/ml_scorer.md`.

## Flood Guard 🌊

Xem `docs/flood_guard.md` để hiểu toàn bộ kiến trúc nhận diện người bị lũ cuốn:
ước lượng dòng chảy bằng optical flow (loại trừ vùng người), đặc trưng
drift_ratio/struggle/submersion, 4 pha WADING → SWEPT / STRANDED / CRITICAL,
và simulator lũ cho demo không cần nước thật.

## Kiến trúc flood / Flood architecture

```
frames ──► SwimDetector (MediaPipe pose) ──► CentroidTracker ──┐
   │                                                           ▼
   └──► FloodFlowEstimator ──► FloodFeatureExtractor ──► FloodRiskEngine ──► Alerts
        (Farnebäck flow,    (drift_ratio, struggle,    (WADING/SWEPT/
         trừ vùng người)     submersion, exposure)      STRANDED/CRITICAL)
```

## Tinh chỉnh / Tuning

Mọi ngưỡng nằm ở `lifestream/config.py` (`RiskConfig.weights`, `calm_frames`,
`danger_frames`, `hysteresis_frames`...). Giảm false positive cho bể trẻ em:
giảm trọng số `vertical_posture`; bể sâu/đông người: tăng `calm_frames`.

## Lộ trình / Roadmap

- [ ] Hiệu chỉnh trên dữ liệu bể bơi/lũ thật + bộ benchmark video gán nhãn
- [ ] Nhận diện vùng nước (pool/water segmentation) để loại bỏ false proposal
- [ ] Multi-camera tracking đồng bộ (đếm người qua biên camera)
- [ ] Tích hợp loa tự động "Ngoài trời, vui lòng ra khỏi nước!"
- [ ] Dashboard lịch sử nhiệt độ rủi ro theo giờ/ngày cho chủ bể
- [ ] Flood Guard: hiệu chuẩn pixels_per_meter tự động từ vật tham chiếu
- [ ] Flood Guard: phát hiện người bị cuốn trên cạn (trượt lở, dòng bùn)
- [ ] ML: fine-tune scorer trên video thật có gán nhãn người; confidence
      per-frame để nới/lồng grace period theo độ tin cậy pose
