import Foundation

// Driver: emits Queryable/ParityFixtures/clip-bpe-tokens-v1.json from the
// real bundled vocab.json / merges.txt. Run with:
//   swiftc -o /tmp/gen_tokens <repo>/Queryable/Queryable/CLIP/Tokenizer/BPETokenizer.swift \
//     <repo>/Queryable/Queryable/CLIP/Tokenizer/BPETokenizer+Reading.swift tools/gen_token_fixtures.swift \
//     -o /tmp/gen_tokens && /tmp/gen_tokens
// Must run from the repository root.

let fm = FileManager.default
let repo = URL(fileURLWithPath: fm.currentDirectoryPath)
let vocabURL = repo.appendingPathComponent("Queryable/Queryable/CoreMLModels/vocab.json")
let mergesURL = repo.appendingPathComponent("Queryable/Queryable/CoreMLModels/merges.txt")

let merges = try BPETokenizer.readMerges(url: mergesURL)
let vocab = try BPETokenizer.readVocabulary(url: vocabURL)
let tokenizer = BPETokenizer(merges: merges, vocabulary: vocab)

struct Case: Encodable {
    let id: String
    let prompt: String
    let paddedLength: Int?
    let expectedTokenIDs: [Int]
}
struct Manifest: Encodable {
    let fixtureVersion: String
    let modelID: String
    let modelRevision: String
    let tokenizerAssets: [String]
    let preprocessingFingerprint: String
    let embeddingDimension: Int
    let contextLength: Int
    let createdAt: String
}
struct File: Encodable {
    let manifest: Manifest
    let cases: [Case]
}

let prompts: [(String, String, Int?)] = [
    ("empty", "", 77),
    ("single-word", "cat", 77),
    ("two-words", "a cat", 77),
    ("punctuation", "Hello, world!", 77),
    ("unicode-diacritics", "café naïve résumé", 77),
    ("emoji", "cat 🐱 dog 🐶", 77),
    ("whitespace", "  multiple   spaces\tand newline\n", 77),
    ("long-description", "a small orange tabby cat sitting on a windowsill next to a potted plant looking outside at the rain", 77),
    ("max-length-truncation", String(repeating: "photograph of a mountain landscape at sunset ", count: 20), 77),
    ("ocr-like", "SALE 50% OFF ends 12/31/2024!!!", 77),
    ("unpadded", "cat", nil),
]

var cases = [Case]()
for (id, prompt, pad) in prompts {
    let (_, ids) = tokenizer.tokenize(input: prompt, minCount: pad)
    cases.append(Case(id: id, prompt: prompt, paddedLength: pad, expectedTokenIDs: ids))
}

let manifest = Manifest(
    fixtureVersion: "clip-bpe-tokens-v1",
    modelID: "mobileclip-s2",
    modelRevision: "mobileclip-s2-v1",
    tokenizerAssets: ["vocab.json", "merges.txt"],
    preprocessingFingerprint: "ci-lanczos-argb-256-v1",
    embeddingDimension: 512,
    contextLength: 77,
    createdAt: "2026-10-08"
)
let file = File(manifest: manifest, cases: cases)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(file)
let outURL = repo.appendingPathComponent("Queryable/ParityFixtures/clip-bpe-tokens-v1.json")
try data.write(to: outURL)
print("wrote \(outURL.path) with \(cases.count) cases, vocab=\(vocab.count)")
