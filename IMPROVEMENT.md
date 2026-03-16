## Những chỗ có thể cải thiện thực sự cho AdvanceHetDDI

### 🔴 Bottleneck #1 — `_make_fc` quá nặng và sai vị trí

```python
# Hiện tại: 3 lớp Linear-BN-Dropout-ReLU TRƯỚC khi có context
nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.Dropout(dropout), nn.ReLU(),  # ×3
```

Ba lớp FC này được áp dụng **độc lập cho từng drug**, trước co-attention. Vấn đề: chúng đang học một projection mà không có context của drug kia. Tệ hơn, BatchNorm1d ở đây sẽ gây vấn đề khi batch size nhỏ hoặc khi test (statistics khác train). **Đề xuất**: giảm xuống 1 lớp FC đơn giản + LayerNorm thay BN.

---

### 🔴 Bottleneck #2 — Co-attention chỉ có `unsqueeze(1)` → sequence length = 1

```python
a_seq = a.unsqueeze(1)  # [B, 1, D]
b_seq = b.unsqueeze(1)  # [B, 1, D]
```

MultiheadAttention với sequence length = 1 thực chất **không làm gì nhiều hơn một phép linear**. Nó tính `softmax(QK^T/√d)` nhưng với chỉ một token thì attention weight luôn = 1.0. Co-attention này không thực sự cross-attend theo nghĩa đúng — nó chỉ là một phép linear transform kèm residual. Đây là vấn đề kiến trúc quan trọng nhất.

**Đề xuất**: Nếu muốn co-attention thực sự, cần dùng **atom-level features** (trước khi pooling), không phải drug-level embeddings đã pool rồi.

---

### 🟡 Bottleneck #3 — Prototype Memory với `τ = 0.1` rất cứng

Temperature 0.1 tạo ra softmax gần như one-hot — prototype nào gần nhất sẽ nhận gần như toàn bộ weight. Điều này làm proto_feat gần như bằng prototype gần nhất, không mang thông tin mới so với logit cuối. Momentum update 0.99 cũng rất chậm.

**Đề xuất**: Thử `τ = 0.5`–`1.0`, hoặc bỏ hẳn prototype memory và thay bằng **label embedding** được concat vào decoder (chỉ dùng trong test time với predicted label từ vòng đầu).

---

### 🟡 Bottleneck #4 — Pair representation là `concat(A, B)` đơn giản

Hiện tại không có interaction term nào giữa A và B trong pair_emb. Co-attention có residual nhưng như đã nêu ở #2, nó yếu. Một interaction đơn giản nhưng hiệu quả:

```python
# Thay vì chỉ concat:
pair_emb = torch.cat([left_emb, right_emb, left_emb * right_emb], dim=-1)
# hoặc thêm absolute difference:
pair_emb = torch.cat([left_emb, right_emb, torch.abs(left_emb - right_emb)], dim=-1)
```

Cả `a*b` và `|a-b|` đều giúp model học **symmetric interactions** mà pure concat bỏ qua.

---

### 🟢 Bottleneck #5 — `kg_fc` và `mol_fc` dùng chung dim

Sau co-attention, kg và mol embeddings được concat mà không có **alignment** giữa hai không gian. `mol_proj` chỉ được dùng cho contrastive loss, không cho fusion. Nên dùng `mol_proj` để đưa mol về cùng không gian với kg **trước** khi concat.

---

### Những gì **không** nên thay đổi

- **HGNN backbone**: Ổn, không phải bottleneck.
- **InfoNCE contrastive**: Tốt, weight 0.05 hợp lý.
- **WarmupCosine scheduler**: Hợp lý.
- **Label smoothing 0.05**: Đúng hướng.
- **Mixup**: Nguyên lý đúng nhưng cần test xem có thực sự giúp với s3 không.

---

## Tóm tắt thứ tự ưu tiên thử

1. **Fix co-attention** — dùng atom-level features, hoặc bỏ đi và thay bằng bilinear interaction đơn giản hơn nhưng đúng hơn về mặt toán học.
2. **Thêm `|a-b|` và `a*b`** vào pair_emb — rủi ro thấp, thường tăng 1–3%.
3. **Giảm `_make_fc`** từ 3 xuống 1 lớp, đổi BN → LayerNorm — giảm overfitting.
4. **Tăng prototype temperature** lên 0.5.