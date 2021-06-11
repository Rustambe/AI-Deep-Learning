from imageai.Detection import VideoObjectDetection # ImageAi kutubxonasini obyekt aniqlash uchun yuklab olamiz
import cv2 # Opencv kutubxonasini web kamerani yoqish uchun yuklab olamiz

camera = cv2.VideoCapture(0)

#second funksiyasi orqali biz ma'lumotlarni to'playmiz va faylga yuboramiz
def second(second_number, output_arrays, count_arrays, average_output_count):
    with open('Database.txt', 'a') as f:
        print("SONIYA : ", second_number, file=f)
        print("Har 3 soniyadagi obyektlar umumiy ma'lumoti ", output_arrays, file=f)
        print("Har soniyadagi obyekt turlari: ", count_arrays, file=f)
        print("Obyektlarning o'rtacha ko'rinishi: ", average_output_count, file=f)
        print("------------SONIYA TAMOM --------------", file=f)

while True:
    frame, img = camera.read()

    detector = VideoObjectDetection() # VideoObjectDetection classini faollashtiramiz
    detector.setModelTypeAsYOLOv3() # YOLOv3 mmodelini o'qishni faollashtiramiz
    detector.setModelPath("./yolo.h5") # Model manzilini kiritamiz
    detector.loadModel(detection_speed="fastest") # Modelni yuklaymiz va tezlikni belgilaymiz

    # detection variabli orqali web kamerani va qayta ishlangan videoni manzilini kiritamiz va qo'shimcha parametrlarni sozlaymiz
    detection = detector.detectObjectsFromVideo(camera_input=camera,
                                                 output_file_path="./camera_detected_video"
                                                 , frames_per_second=3, per_second_function=second,
                                                 minimum_percentage_probability=30)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()






