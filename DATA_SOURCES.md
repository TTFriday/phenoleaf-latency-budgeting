# Data and model sources

No source images, masks, label maps, model checkpoints or per-frame
prediction streams are redistributed in this repository.  The archived
outputs are aggregate metrics, timing records, controller features and
verification reports computed from the development roles.

## PhenoLeaf-TS

`shared/pilot_inventory.csv` is a portable copy of the audited inventory.
Its image and label paths are relative to a separately obtained
PhenoLeaf-TS data root.  Pixel hashes, roles, timestamps, instance counts
and quarantine annotations are unchanged.  The original inventory SHA-256
and the published SHA-256 are recorded in
`shared/inventory_provenance.json`.

The upstream dataset licence restricts redistribution of dataset files.
Obtain the media from the original source and comply with its licence and
citation requirements.  Locked roles (`C_POLICY`, `T_ID`, `T_GENO`, `T_EXT`)
are metadata only in this package and were not used for model selection or
effect analysis.

## COCO initialization

Detector runs use the COCO Mask R-CNN ResNet-50-FPN initialization shipped
by torchvision.  The expected filename is
`maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`; set `PHENOLEAF_COCO_WEIGHTS` or pass
`--weights` to select its location.  The checkpoint is not included.
