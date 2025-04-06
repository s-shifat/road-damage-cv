# Road Damage Detection Using CV

## Setup

- Clone this repository and cd into it
    ```bash
    git clone https://github.com/s-shifat/road-damage-cv.git 
    cd road-damage-cv
    ```

-  Create a virtual environment
    ```bash
    python -m venv venv
    ```
-  Activate the environment
    * windows:
        ```powershell
        .\venv\Scripts\activate.ps1
        ```
    * linux/MacOS:
        ```powershell
        source ./venv/bin/activate
        ```

- Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```
- Optional: To list the index of available webcams:
    ```bash
    python main.py list
    ```
 
- Run the script
    ```bash
    python main.py run ./best.pt
    ```

    *Optionally pass `-r` flag to record the video and set webcam device index using `-wc`*
    For Example:
    ```bash
    python main.py run -r -wc 0 ./best.pt
    ```
    This will capture the video using webcam device at index 0 and save the video in disk.


## Help Pages

```bash
python main.py -h
```

```text
usage: main.py [-h] {list,run} ...

positional arguments:
  {list,run}
    list      List available index of webcam devices
    run       Run the model on webcam

options:
  -h, --help  show this help message and exit
```


---

```bash
python main.py run -h
```
```text
usage: main.py run [-h] [-wc WEBCAM] [-r] model_path

positional arguments:
  model_path            Path to the model

options:
  -h, --help            show this help message and exit
  -wc, --webcam WEBCAM  Webcam device index
  -r, --record          If passed, the video will be recorded and saved to
                        ./recorded_videos/

```
