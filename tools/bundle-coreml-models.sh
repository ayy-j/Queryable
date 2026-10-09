#!/bin/sh
# Package the default model from persistent checkout files, never Derived Data.
set -eu

resource_source="${SRCROOT}/Queryable/CoreMLModels"
model_cache="${SRCROOT}/../models"
resource_destination="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/CoreMLModels"

# Resolve and validate every required artifact before changing the app bundle.
for artifact in vocab.json merges.txt; do
    if [ ! -s "${resource_source}/${artifact}" ]; then
        echo "error: Missing tokenizer asset ${resource_source}/${artifact}. Restore it from the repository."
        exit 1
    fi
done

find_model() {
    for model_directory in "${resource_source}/$1" "${model_cache}/$1"; do
        if [ -s "${model_directory}/coremldata.bin" ] &&
           [ -s "${model_directory}/model.mil" ] &&
           [ -s "${model_directory}/weights/weight.bin" ]; then
            printf '%s\n' "${model_directory}"
            return 0
        fi
    done
    echo "error: Missing or incomplete $1. Place the complete compiled model in ${resource_source} or ${model_cache}, then rebuild." >&2
    return 1
}

image_source="$(find_model ImageEncoder_mobileCLIP2_s4.mlmodelc)"
text_source="$(find_model TextEncoder_mobileCLIP2_s4.mlmodelc)"

mkdir -p "${resource_destination}"
for artifact in vocab.json merges.txt; do
    /usr/bin/rsync -a "${resource_source}/${artifact}" "${resource_destination}/"
done
for model_source in "${image_source}" "${text_source}"; do
    model_name="${model_source##*/}"
    /usr/bin/rsync -a --delete --exclude=.DS_Store "${model_source}/" "${resource_destination}/${model_name}/"
done

echo "Bundled MobileCLIP2-S4 image/text models and tokenizer assets."
