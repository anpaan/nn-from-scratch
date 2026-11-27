# TASKS: Neural Network from Scratch

Lộ trình chia thành 10 milestone. Mỗi milestone gồm:
- 📖 **Kiến thức**: đọc gì trong `docs/notes.md` trước khi code
- 🛠 **Việc cần làm**: checklist
- ✅ **Xong khi**: tiêu chí kiểm tra

**Quy ước test:** chạy `python -m pytest -v` ở thư mục gốc.
`SKIPPED` = chưa viết · `FAILED` = viết rồi nhưng sai · `PASSED` = xong.

**Quy tắc học:** tự viết phần `TODO` trước. Bí quá thì mở phần 💡 *Gợi ý*, hoặc hỏi Claude giải thích chứ đừng xin code ngay.

---

## Phần 1: Engine (Python + NumPy)

### M0. Setup
📖 Không cần.
🛠
- [ ] Tạo virtualenv: `python -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] `python -m pytest -v`: thấy toàn bộ test **SKIPPED** (chưa có FAILED nào) là đúng
- [ ] `git init`, commit đầu tiên, tạo repo `nn-from-scratch` trên GitHub, push lên
- [ ] Vào tab **Actions** trên GitHub, xem CI chạy xanh

✅ Repo đã lên GitHub, CI xanh.

---

### M1. Linear layer + activations
📖 notes.md mục 1, 2, 3, 6
🛠
- [ ] Đọc kỹ `ReLU` trong `engine/activations.py`. Đây là ví dụ mẫu đã viết sẵn: forward cache lại cái gì, backward dùng nó ra sao.
- [ ] `Linear.forward` và `Linear.backward` (`engine/layers.py`)
- [ ] `Sigmoid` và `Tanh` (forward + backward)
- [ ] Đọc `engine/gradcheck.py` để hiểu vì sao test so được gradient của anh với gradient "đo" bằng số

<details><summary>💡 Gợi ý Linear.backward</summary>

Chỉ cần ghép shape là ra:
- `dW` phải có shape `(in, out)`. Từ `x (batch, in)` và `grad_out (batch, out)` thì chỉ có `x.T @ grad_out` khớp.
- `db` có shape `(out,)`: cộng `grad_out` theo trục batch, tức `grad_out.sum(axis=0)`.
- `dx` có shape `(batch, in)`: `grad_out @ W.T`.

Vì sao lại như vậy? Mỗi phần tử `W[i,j]` ảnh hưởng tới output cột j của **mọi** mẫu trong batch, nên gradient phải cộng dồn qua batch, và phép nhân ma trận làm đúng việc đó.
</details>

✅ `tests/test_layers.py` PASSED hết.

---

### M2. MSE, SGD, Sequential
📖 notes.md mục 4, 5, 7
🛠
- [ ] `MSE.forward` / `MSE.backward` (`engine/losses.py`)
- [ ] `SGD.step`: cập nhật **in-place** (`p -= lr * g`), đọc kỹ cảnh báo trong code
- [ ] `Sequential.forward` / `Sequential.backward` (`engine/network.py`)

<details><summary>💡 Vì sao phải in-place?</summary>

`p = p - lr*g` tạo ra một mảng **mới** và gán vào biến cục bộ `p`, còn `layer.W` vẫn trỏ vào mảng cũ, nên mạng không học gì cả. `p -= lr*g` thì sửa trực tiếp vùng nhớ mà `layer.W` đang trỏ tới. Giống con trỏ trong C: sửa `*p` khác với sửa `p`.

Cùng một lý do, trong `Linear.backward` phải viết `self.dW[...] = ...` chứ không phải `self.dW = ...`. Optimizer được tạo một lần với danh sách tham chiếu tới các mảng gradient. Nếu gán mảng mới thì optimizer vẫn đọc mảng cũ (toàn số 0), nên test gradient vẫn PASSED nhưng mạng lại không học. Đây là loại bug "im lặng" khó tìm nhất.
</details>

✅ `test_mse_*` PASSED.

---

### M3. Train XOR, mạng đầu tiên học được 🎉
📖 notes.md mục 3 (đọc lại vì sao XOR cần activation)
🛠
- [ ] Bài này cần softmax CE ở M4. Có thể làm M4 trước, hoặc tạm sửa `examples/xor.py` sang dùng `MSE` + `Sigmoid` với target dạng `y.reshape(-1, 1)` và 1 đầu ra
- [ ] Chạy `python -m examples.xor`, xem loss giảm dần
- [ ] Thí nghiệm: bỏ `Tanh()` đi rồi chạy lại. Mạng có học được không? Vì sao?
- [ ] Thí nghiệm: đổi lr thành 5.0, rồi 0.001. Ghi nhận xét vào notes.md

✅ Mạng đoán đúng cả 4 điểm XOR.

---

### M4. Softmax + Cross-Entropy
📖 notes.md mục 8
🛠
- [ ] `SoftmaxCrossEntropy.forward` / `backward`
- [ ] Nhớ mẹo trừ max để không bị tràn số (có test riêng cho chuyện này)

<details><summary>💡 Tạo one-hot nhanh</summary>

`one_hot = np.eye(n_classes)[target]`, hoặc tránh tạo hẳn ma trận: copy `probs`, rồi `grad[np.arange(batch), target] -= 1`.
</details>

✅ Toàn bộ `tests/test_losses.py` và `tests/test_training.py` PASSED.

---

### M5. Adam + Spiral + vẽ ranh giới quyết định
📖 Tự tìm hiểu "momentum", có thể hỏi Claude giải thích Adam
🛠
- [ ] `Adam.step` với công thức (mỗi tham số, `t` tăng 1 mỗi lần step):
  ```
  m = β1·m + (1-β1)·g          # trung bình động của gradient (quán tính)
  v = β2·v + (1-β2)·g²         # trung bình động của gradient²
  m̂ = m / (1-β1^t)   v̂ = v / (1-β2^t)     # bias correction
  p -= lr · m̂ / (√v̂ + ε)
  ```
- [ ] Viết vòng train trong `examples/spiral.py`, chạy và mở `docs/spiral.png`
- [ ] So sánh SGD với Adam trên spiral: cái nào hội tụ nhanh hơn?

✅ Accuracy spiral trên 95%, có ảnh ranh giới xoắn đẹp. Ảnh này dùng cho README!

---

### M6. MNIST: nhận diện chữ số viết tay
📖 notes.md mục 9
🛠
- [ ] Tải `mnist.npz` (ví dụ bản của Keras: `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`), đặt vào `data/`. Thư mục này đã nằm trong .gitignore
- [ ] Viết `load_mnist()`: các key là `x_train, y_train, x_test, y_test`; reshape về `(N, 784)` và chia 255
- [ ] Train theo **mini-batch** trong `examples/mnist.py`, shuffle lại mỗi epoch
- [ ] In test accuracy sau mỗi epoch
- [ ] Lưu weights ra file để dùng ở M9

✅ Test accuracy ≥ 97%.

---

## Phần 2: Playground (TypeScript, chạy trên trình duyệt)

### M7. Engine TypeScript + heatmap ranh giới
📖 Học nhanh TypeScript cơ bản: type, class, interface. Hỏi Claude nếu cần
🛠
- [ ] Cài Node.js, rồi `cd playground && npm install && npm run dev`
- [ ] Viết lại engine nhỏ trong `src/engine.ts`: Linear, Tanh, ReLU, SoftmaxCE, Sequential, SGD
- [ ] Port các dataset 2D (spiral, circles, XOR) sang TS
- [ ] Vòng lặp `requestAnimationFrame`: mỗi frame train vài bước, rồi tô heatmap lên canvas

✅ Mở trình duyệt thấy ranh giới "mọc" dần quanh dữ liệu trong lúc train.

---

### M8. Vẽ mạng + bảng điều khiển + biểu đồ loss
🛠
- [ ] Vẽ mạng bằng SVG: nơ-ron là các chấm tròn, đường nối dày theo `|w|`, màu theo dấu (xanh +, cam −)
- [ ] Controls: chọn dataset, learning rate, số lớp và nơ-ron, activation, nút Play / Pause / Reset
- [ ] Biểu đồ loss theo thời gian
- [ ] Tuỳ chọn: hover vào nơ-ron để xem heatmap riêng của nơ-ron đó

✅ Chỉnh thông số là mạng reset và học lại ngay.

---

### M9. Vẽ chữ số để mạng đoán
🛠
- [ ] Viết script Python export weights MNIST ra JSON (`playground/public/mnist.json`)
- [ ] Canvas 280×280 cho người dùng vẽ, downscale về 28×28 rồi đưa qua mạng
- [ ] Hiển thị xác suất của 10 chữ số dạng thanh bar
- [ ] Lưu ý: MNIST có chữ số căn giữa và nét dày. Nếu đoán sai nhiều, thử căn giữa hình vẽ theo tâm khối

✅ Vẽ số 7, mạng đoán ra 7.

---

### M10. Deploy + README
🛠
- [ ] GitHub Actions deploy `playground/dist` lên **GitHub Pages**
- [ ] README: GIF demo, link playground, ảnh spiral, phần "What I learned"
- [ ] Badge CI ở đầu README
- [ ] Tag `v1.0.0` 🎉

✅ Có link public để khoe.

---

## Ý tưởng sau v1 (tuỳ chọn)
- Autograd kiểu micrograd: tự động hoá backward bằng computational graph
- Thêm Dropout và L2 regularization, xem tác động lên overfitting
- Lớp Convolution (CNN) cho MNIST, mục tiêu trên 99%
