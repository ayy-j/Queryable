#!/bin/bash
# Run the production GPU index and its regression tests natively, without UIKit.
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo "This test runner requires macOS and Metal." >&2
    exit 1
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
package_root="$(mktemp -d /private/tmp/queryable-gpu-mutations.XXXXXX)"
trap 'rm -rf "$package_root"' EXIT
mkdir -p "$package_root/Sources/GPUIndexMutations" "$package_root/Tests/GPUIndexMutationsTests"
cp "$repo_root/Queryable/Queryable/Model/GPUSimilaritySearch.swift" "$package_root/Sources/GPUIndexMutations/"
sed 's/@testable import Queryable/@testable import GPUIndexMutations/' \
    "$repo_root/Queryable/QueryableTests/SimilarityValidationTests.swift" \
    > "$package_root/Tests/GPUIndexMutationsTests/SimilarityValidationTests.swift"
cat > "$package_root/Package.swift" <<'PACKAGE'
// swift-tools-version: 5.9
import PackageDescription
let package = Package(
    name: "GPUIndexMutations",
    platforms: [.macOS(.v14)],
    products: [.library(name: "GPUIndexMutations", targets: ["GPUIndexMutations"])],
    targets: [
        .target(name: "GPUIndexMutations"),
        .testTarget(name: "GPUIndexMutationsTests", dependencies: ["GPUIndexMutations"])
    ]
)
PACKAGE
swift test --package-path "$package_root" --scratch-path "$package_root/build" "$@"
