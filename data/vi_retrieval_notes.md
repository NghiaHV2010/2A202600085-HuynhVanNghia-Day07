# Ghi chú về Retrieval cho Trợ lý Tri thức Nội bộ

Trong một hệ thống trợ lý tri thức nội bộ, retrieval đóng vai trò tìm ra những đoạn tài liệu phù hợp nhất trước khi mô hình ngôn ngữ tạo câu trả lời. Mục tiêu không chỉ là trả lời trôi chảy mà còn phải bám sát nguồn dữ liệu đã được lưu trữ.

Một pipeline retrieval điển hình bắt đầu từ việc thu thập tài liệu, làm sạch nội dung, chia nhỏ thành các chunk có ý nghĩa, sau đó tạo embedding cho từng chunk. Khi người dùng đặt câu hỏi, hệ thống sẽ tạo embedding cho câu hỏi đó và so sánh với các vector đã lưu để tìm các đoạn gần nhất về mặt ngữ nghĩa.

Chất lượng chunking ảnh hưởng trực tiếp đến chất lượng retrieval. Nếu chunk quá ngắn, hệ thống có thể trả về các câu rời rạc thiếu ngữ cảnh. Nếu chunk quá dài, nhiều ý không liên quan sẽ bị gộp lại, làm giảm độ chính xác của kết quả. Vì vậy, nhiều nhóm chọn chiến lược recursive chunking để ưu tiên tách theo đoạn, rồi mới tách nhỏ hơn khi cần.

Metadata cũng rất quan trọng. Ví dụ, một công ty có thể gắn nhãn tài liệu theo phòng ban, ngôn ngữ, độ nhạy cảm, hoặc ngày cập nhật. Khi người dùng hỏi về tài liệu kỹ thuật bằng tiếng Việt, bộ lọc metadata có thể giúp hệ thống tránh lấy nhầm các tài liệu marketing hoặc tài liệu tiếng Anh không liên quan.

Trong thực tế, retrieval không phải lúc nào cũng đúng. Một số lỗi thường gặp là tài liệu cũ vẫn xếp hạng cao, từ khóa trong câu hỏi không khớp với cách diễn đạt trong tài liệu, hoặc embedding model chưa xử lý tốt nội dung song ngữ. Vì vậy, đội ngũ phát triển nên kiểm thử bằng các truy vấn thực tế, xem trực tiếp các chunk được trả về, và ghi nhận failure cases để cải thiện dữ liệu cũng như chiến lược truy xuất.
