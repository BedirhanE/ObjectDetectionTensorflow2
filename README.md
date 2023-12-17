# TensorFlow  Object Detection

Bu proje, TensorFlow Hub modülü kullanarak nesne tespiti yapmayı amaçlayan bir örnektir.

## Giriş

Bu örnek, TensorFlow ve TensorFlow Hub kullanılarak oluşturulmuştur. TensorFlow Hub, önceden eğitilmiş modülleri kolayca entegre etmenizi sağlar.

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki gereksinimlere ihtiyacınız vardır:

- TensorFlow
- TensorFlow Hub
- Matplotlib
- Pillow

Gerekli kütüphaneleri yüklemek için aşağıdaki komutları kullanabilirsiniz:

```bash
pip install tensorflow tensorflow-hub matplotlib Pillow

## Kullanım

1. `display_image(image)` fonksiyonu ile bir resmi ekranda gösterme.
2. `download_and_resize_image(url, new_width=256, new_height=256, display=False)` fonksiyonu ile bir resmi indirme ve yeniden boyutlandırma.
3. `draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=())` fonksiyonu ile bir resim üzerine sınırlayıcı kutular çizme.
4. `draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1)` fonksiyonu ile resim üzerine etiketli kutular ekleyerek sonuçları görselleştirme.

Bu kodlar, nesne tespiti sonuçlarını görselleştirmek ve anlamak için kullanışlıdır.

## Örnek Kullanım

```python
# Örnek kullanım
image_url = "https://example.com/image.jpg"
image_path = download_and_resize_image(image_url, display=True)

# Önceden eğitilmiş modelden nesne tespiti yapma
# (Kod örnekleri buraya eklenmeli)

# Sonuçları görselleştirme
image = Image.open(image_path)
# (Kod örnekleri buraya eklenmeli)
display_image(image)


# Örnek Kullanım: Resim İndirme ve Boyutlandırma

Bu bölümde, belirli bir URL'den resim indirme ve boyutlandırma işlemleri açıklanmaktadır. 

## Kullanım

Aşağıdaki örnek, belirtilen bir URL'den bir resmi indirip boyutlandırarak görselleştirir:

```python
# By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg"
downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)


# Örnek Kullanım: TensorFlow Hub Faster R-CNN Nesne Tespiti

Bu bölümde, TensorFlow Hub üzerinden Faster R-CNN modeli kullanarak nesne tespiti yapma işlemleri açıklanmaktadır.

## Kullanım

Aşağıdaki kod örneği, TensorFlow Hub üzerinden Faster R-CNN modelini yükler, belirtilen bir resim dosyasını analiz eder ve nesneleri tespit ederek görselleştirir:

```python
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(detector, path):
  img = load_img(path)

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key:value.numpy() for key,value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time-start_time)

  image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

  display_image(image_with_boxes)

# Örnek kullanım
run_detector(detector, downloaded_image_path)

Bu kod parçası, belirtilen Faster R-CNN modelini TensorFlow Hub'dan yükler ve bir resim dosyasını analiz ederek nesne tespiti yapar. Sonuçları ekranda görselleştirir.

#Detaylar
Bu örnek, TensorFlow Hub kullanarak bir önceden eğitilmiş Faster R-CNN modelini yükleyip kullanmayı gösterir. run_detector fonksiyonu, belirtilen resim üzerinde nesne tespiti yapar, sonuçları ekrana yazdırır ve görselleştirir.

# Örnek Kullanım: Birden Fazla Resim Üzerinde Nesne Tespiti

Bu bölümde, TensorFlow Hub Faster R-CNN modelini kullanarak birden fazla resim üzerinde nesne tespiti yapma işlemleri açıklanmaktadır.

## Kullanım

Aşağıdaki kod örneği, belirtilen bir dizi resim URL'si üzerinde Faster R-CNN modeli ile nesne tespiti yapar ve sonuçları görselleştirir:

```python
image_urls = [
  "https://upload.wikimedia.org/wikipedia/commons/1/1b/The_Coleoptera_of_the_British_islands_%28Plate_125%29_%288592917784%29.jpg",
  "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg/1024px-Biblioteca_Maim%C3%B3nides%2C_Campus_Universitario_de_Rabanales_007.jpg",
  "https://upload.wikimedia.org/wikipedia/commons/0/09/The_smaller_British_birds_%288053836633%29.jpg",
  "https://m.media-amazon.com/images/I/61v9vPMuk6L._AC_SX679_.jpg",
  "https://pixnio.com/free-images/2017/06/08/2017-06-08-13-07-49-1152x867.jpg",
]

def detect_img(image_url):
  start_time = time.time()
  image_path = download_and_resize_image(image_url, 640, 480)
  run_detector(detector, image_path)
  end_time = time.time()
  print("Inference time:", end_time - start_time)

# Örnek kullanım
detect_img(image_urls[0])
detect_img(image_urls[1])
detect_img(image_urls[2])
detect_img(image_urls[3])
detect_img(image_urls[4])


Bu kod parçası, belirtilen resim URL'leri üzerinde Faster R-CNN modelini kullanarak nesne tespiti yapar ve sonuçları ekranda görselleştirir

#Detaylar
Bu örnek, bir dizi resim URL'si üzerinde döngü kullanarak nesne tespiti yapma sürecini gösterir.
Her bir resmin nesne tespiti sonuçları, ayrı ayrı ekranda görselleştirilir.

#Not
Bu örnek, TensorFlow Hub üzerinden alınan bir modeli kullanarak sadece öğrenme amacıyla tasarlanmıştır. Kodu inceleyerek ve özelleştirerek kendi projenize uyarlayabilirsiniz.

