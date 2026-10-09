import CoreGraphics

/// The native benchmark feeds generated CGImages to the unchanged encoding
/// pipeline. UIKit's only image API used by that pipeline is `cgImage`.
public struct UIImage {
    public let cgImage: CGImage?
    public init(cgImage: CGImage) { self.cgImage = cgImage }
}
