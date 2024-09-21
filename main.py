from ultralytics import YOLO
import cv2
import os

# Eğitilmiş YOLOv8 modelinizi yükleyin
model = YOLO("")  # Kendi eğitilmiş model dosyanızı kullanın(best.pt)

# Tahmin yapacak fonksiyon
def predict(image_path):
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    # Görüntüyü yükleyin
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load the image '{image_path}'.")
        return

    # YOLOv8 ile tahmin yapın
    results = model(image)

    # Sonuçları işleyin ve görüntüye çizin
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # Kutuları ve etiketleri çiz
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Kırmızı renk (BGR formatında)

            # Metni kırmızı renkte çiz
            cv2.putText(image, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)  # Kırmızı renk (BGR formatında)

    return image  # İşlenmiş görüntüyü döndürür

# Klasördeki tüm görüntüler için tahmin yapma
input_folder = ""  # Imageların olduğu klasör yolu
output_folder = ""  # Kaydedilecek klasör yolu

# Çıkış klasörünü oluştur
os.makedirs(output_folder, exist_ok=True)

# Klasördeki her bir görüntüyü işle
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"output_{filename}")  # Yeni dosya adı

        # Tahmin yap ve çıktıyı kaydet
        processed_image = predict(image_path)
        cv2.imwrite(output_path, processed_image)
        print(f"Output saved to {output_path}")

print("All images processed.")