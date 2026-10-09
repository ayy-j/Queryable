// swift-tools-version: 5.9
import PackageDescription

// The Python runner stages the actual application sources beside this manifest.
// No package dependencies, model downloads, or signed app host are needed.
let package = Package(
    name: "QueryablePerformance",
    platforms: [.macOS(.v14)],
    targets: [
        .target(name: "Queryable"),
        .testTarget(name: "NativePerformanceTests", dependencies: ["Queryable"])
    ]
)
