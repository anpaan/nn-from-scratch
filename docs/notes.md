# Sổ tay khái niệm: Neural Network cơ bản

Đây là sổ tay để đọc song song với `TASKS.md`. Mỗi milestone sẽ trỏ về mục nào cần đọc. Nhớ ghi thêm ghi chú của riêng anh vào đây trong lúc học.

---

## 1. Một nơ-ron làm gì?

Một nơ-ron nhận vài con số đầu vào, nhân mỗi số với một **trọng số (weight)**, cộng tất cả lại, cộng thêm một **bias**:

```
z = w1*x1 + w2*x2 + ... + b
```

Sau đó nó đưa `z` qua một **hàm kích hoạt (activation)**, ví dụ ReLU: `a = max(0, z)`.

Cách hình dung quen thuộc với dân embedded: nó giống một bộ cộng có trọng số rồi qua một bộ so ngưỡng "mềm". Weight cho biết đầu vào nào quan trọng, bias dịch ngưỡng lên hoặc xuống.

"Học" nghĩa là **tìm giá trị cho các w và b** sao cho đầu ra đúng.

## 2. Từ một nơ-ron đến một lớp, rồi đến cả mạng

Một **lớp (layer)** gồm nhiều nơ-ron cùng nhìn chung một đầu vào. Nếu viết từng nơ-ron bằng vòng `for` thì rất chậm, nên ta gom lại thành ma trận:

```
Y = X @ W + b
X: (batch, in)    W: (in, out)    b: (out,)    Y: (batch, out)
```

- Mỗi **hàng** của X là một mẫu dữ liệu. Xử lý nhiều mẫu một lúc gọi là **batch**.
- Mỗi **cột** của W là trọng số của một nơ-ron.

Một **mạng (network)** là các lớp xếp nối tiếp nhau: `Linear → ReLU → Linear → ReLU → Linear`.

> Mẹo sống còn: luôn **ghi shape** bên cạnh code. 90% bug ở giai đoạn đầu là sai shape.

## 3. Vì sao cần hàm kích hoạt?

Nếu chỉ chồng các lớp Linear lên nhau: `(X @ W1) @ W2 = X @ (W1 @ W2)`, kết quả vẫn chỉ là **một** phép biến đổi tuyến tính. Mạng sâu bao nhiêu thì cũng chỉ vẽ được **đường thẳng** để phân chia dữ liệu.

Chèn ReLU/Tanh vào giữa các lớp sẽ "bẻ cong" không gian, nhờ vậy mạng vẽ được ranh giới cong. Đó là lý do bài XOR (M3) bắt buộc phải có activation.

| Hàm     | Công thức          | Đạo hàm          | Dùng khi                          |
|---------|--------------------|------------------|-----------------------------------|
| ReLU    | max(0, x)          | 1 nếu x>0, còn lại 0 | Mặc định cho lớp ẩn            |
| Sigmoid | 1/(1+e^-x)         | s(1-s)           | Cần đầu ra trong (0,1)           |
| Tanh    | tanh(x)            | 1 - tanh²        | Mạng nhỏ, dữ liệu 2D               |

## 4. Loss: đo mức độ sai

**Loss** là *một con số duy nhất* cho biết mạng sai bao nhiêu. Loss càng nhỏ càng tốt.

- **MSE**, cho bài toán hồi quy (đoán một con số): `mean((pred - target)²)`.
- **Cross-entropy**, cho bài toán phân loại (đoán nhãn): xem mục 8.

Toàn bộ việc train chỉ là: **chỉnh w, b để loss giảm**.

## 5. Gradient và gradient descent

**Gradient** `∂Loss/∂w` trả lời câu hỏi: *"Nếu tăng w lên một chút xíu thì loss tăng hay giảm, và nhiều cỡ nào?"*

Hình dung anh đứng trên đồi trong sương mù và muốn xuống thung lũng. Anh không thấy đường, nhưng cảm được dốc dưới chân, nên cứ bước một bước nhỏ về phía dốc xuống. Đó là **gradient descent**:

```
w = w - learning_rate * gradient
```

Dấu trừ có nghĩa là đi **ngược** hướng làm loss tăng.

## 6. Chain rule và backpropagation (phần quan trọng nhất)

Mạng là một chuỗi hàm lồng nhau: `loss = L(f3(f2(f1(x))))`. **Chain rule** (quy tắc đạo hàm hàm hợp) cho phép tính đạo hàm của cả chuỗi bằng cách **nhân các đạo hàm từng khâu**:

```
∂loss/∂x = ∂loss/∂f3 · ∂f3/∂f2 · ∂f2/∂f1 · ∂f1/∂x
```

Ví dụ tay: `y = (2x + 1)²`, đặt `u = 2x + 1`.
- `dy/du = 2u`, `du/dx = 2`, suy ra `dy/dx = 2u · 2 = 4(2x+1)`.
- Tại x = 1: u = 3, dy/dx = 12. ✔

**Backpropagation** chính là chain rule chạy **từ cuối về đầu**:
1. Loss đưa ra `grad` (∂loss/∂output) cho lớp cuối.
2. Mỗi lớp nhận `grad_out`, tính gradient cho tham số của nó (dW, db), rồi trả về `grad_in = ∂loss/∂input` cho lớp phía trước.
3. Lặp lại cho đến lớp đầu tiên.

Vì vậy mỗi lớp trong repo có `forward()` và `backward()`. Lớp nào cũng phải **cache** input hoặc output trong forward, vì backward cần đến chúng.

Liên hệ embedded: nó giống truyền tín hiệu lỗi ngược qua một pipeline, mỗi khâu biết "độ nhạy" (gain) của riêng nó và nhân vào tín hiệu đó.

## 7. Learning rate

- Quá lớn: bước quá dài, nhảy qua thung lũng, loss dao động hoặc tăng lên vô cùng (`nan`).
- Quá nhỏ: học rất chậm.
- Thường thử: 0.1, 0.01, 0.001. Anh sẽ chỉnh thanh trượt learning rate trong playground (M8) và thấy tận mắt.

## 8. Softmax và cross-entropy

Lớp cuối của bài phân loại cho ra các **logits**, tức điểm thô cho từng lớp, ví dụ `[2.0, 0.5, -1.0]`.

**Softmax** biến logits thành xác suất cộng lại bằng 1:
```
p_i = e^(z_i) / Σ e^(z_j)
```

**Cross-entropy** phạt khi xác suất của nhãn đúng thấp: `loss = -log(p_đúng)`.
- p_đúng = 1 thì loss = 0.  p_đúng = 0.01 thì loss ≈ 4.6.

Ghép hai cái lại thì gradient cực gọn: `∂loss/∂z = p - one_hot(nhãn)` (chia thêm batch size nếu lấy mean).

**Mẹo ổn định số:** `e^1000` bị tràn số. Trừ max của mỗi hàng trước khi tính `e^z` sẽ không đổi kết quả softmax nhưng an toàn hơn.

## 9. Train / test và overfitting

- **Train set** là dữ liệu để học. **Test set** là dữ liệu mạng *chưa từng thấy*, dùng để chấm điểm thật.
- **Overfitting** là khi mạng "học thuộc lòng" train set: train accuracy 100% nhưng test thấp. Giống học tủ đề cương.
- **Epoch** = một lần đi hết train set. **Mini-batch** = chia train set thành các nhóm nhỏ (ví dụ 64 mẫu) và cập nhật sau mỗi nhóm.

## 10. Từ điển nhanh

| Thuật ngữ | Nghĩa |
|---|---|
| weight / bias | tham số cần học |
| forward pass | tính đầu ra từ đầu vào |
| backward pass / backprop | tính gradient từ loss ngược về |
| gradient | độ nhạy của loss theo một tham số |
| learning rate | độ dài mỗi bước cập nhật |
| epoch | một vòng qua toàn bộ dữ liệu train |
| batch | nhóm mẫu xử lý cùng lúc |
| logits | điểm thô trước softmax |
| optimizer | thuật toán cập nhật tham số (SGD, Adam) |
| overfitting | học thuộc train, kém trên dữ liệu mới |
