import UIKit
import CoreML
import Vision // 이미지 고급 처리
import ImageIO

// 시뮬레이터는 결과가 엉터리로 나옴
// 반드시 실제 기기에서 테스트

class ImageClassificationViewController: UIViewController {
    // MARK: - IBOutlets
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var cameraButton: UIBarButtonItem!
    @IBOutlet weak var classificationLabel: UILabel!
    
    /// - Tag: MLModelSetup
    lazy var classificationRequest: VNCoreMLRequest = {
        do {
            // 모델 파일에 접근할 수 있도록 인스턴스 생성
            let model = try VNCoreMLModel(for: HorsesOrHumansClassifier(configuration: MLModelConfiguration()).model)
            
            // Core ML 리퀘스트 인스턴스
            // CompletionHandler: 모델 초기화가 완료되면 처리할 내용
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.processClassifications(for: request, error: error)
            })
            
            // 이미지 분류를 요청할 때, 이미지가 크거나 비율이 다를 경우 이미지를 어디서 취할 것인가?
            // .centerCrop이 가장 많이 사용됨
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 카메라를 이용할 수 없으면 카메라 버튼 비활성화
        if !UIImagePickerController.isSourceTypeAvailable(.camera) {
            cameraButton.isEnabled = false
        }
    }
    
    // MARK: - Photo Actions
    
    @IBAction func takePicture() {
        // Show options for the source picker only if the camera is available.
        // 카메라를 사용할 수 있는 경우에만 source picker에 대한 옵션을 표시합니다.
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            return
        }
        
        presentPhotoPicker(sourceType: .camera)
    }
    
    @IBAction func choosePicture(_ sender: UIBarButtonItem) {
        self.presentPhotoPicker(sourceType: .photoLibrary)
    }
    
    func presentPhotoPicker(sourceType: UIImagePickerController.SourceType) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = sourceType
        present(picker, animated: true)
    }
}

extension ImageClassificationViewController {
    
    // MARK: - Image Classification
    
    /// - Tag: PerformRequests
    func updateClassifications(for image: UIImage) {
        classificationLabel.text = "Classifying..."
        
        guard let orientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue)), // 이미지 방향
                let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                /*
                 이 핸들러는 일반적인 이미지 처리 오류를 포착합니다. `classificationRequest`의 완료 핸들러 `processClassifications(_:error:)`는 해당 요청 처리와 관련된 오류를 포착합니다.
                 */
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }
    
    /// 분류 결과를 바탕으로 UI를 업데이트합니다.
    /// - Tag: ProcessClassifications
    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            guard let results = request.results else {
                self.classificationLabel.text = "Unable to classify image.\n\(error!.localizedDescription)"
                return
            }

            // `result`는 이 프로젝트의 Core ML 모델에서 지정한 대로 항상 'VNClassificationObservation'이 됩니다.
            let classifications = results as! [VNClassificationObservation]
        
            if classifications.isEmpty {
                self.classificationLabel.text = "Nothing recognized."
            } else {
                let topClassifications = classifications.prefix(2)
                let descriptions = topClassifications.map { classification in
                    // Formats the classification for display; e.g. "(0.37) cliff, drop, drop-off".
                    // 표시할 분류 형식을 지정합니다. 예) "(0.37) Dog".
                   return String(format: "  (%.2f) %@", classification.confidence, classification.identifier)
                }
                self.classificationLabel.text = "Classification:\n" + descriptions.joined(separator: "\n")
            }
        }
    }
}

extension ImageClassificationViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    // MARK: - Handling Image Picker Selection

    private func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String: Any]) {
        picker.dismiss(animated: true)
        
        // `imagePickerController(:didFinishPickingMediaWithInfo:)`는 원본 이미지를 제공할 것입니다.
        let image = info[UIImagePickerController.InfoKey.originalImage.rawValue] as! UIImage
        imageView.image = image
        updateClassifications(for: image)
    }
}
