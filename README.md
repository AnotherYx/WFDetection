# End-to-End Multi-Tab Website Fingerprinting Attack: A Detection Perspective


## Getting Started

- Install `cvpods` and `pycocotools`
  ```shell
  cd WFDetection/
  python setup.py develop
  ```
  
- Train
    ```shell
    cd WFDetection/WFDplayground
    pods_train --num-gpus 1
    ```
    
- Test
    ```shell
    cd WFDetection/WFDplayground
    pods_test --num-gpus 1 MODEL.WEIGHTS WFDetection/WFDplayground/output/model_final.pth
    ```


## Citation

    @citing{
      title={End-to-End Multi-Tab Website Fingerprinting Attack: A Detection Perspective},
      year={2022}
    }