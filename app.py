import os
import cv2
import pickle
import uvicorn
# import pytesseract
import numpy as np
import tensorflow as tf

from PIL import Image
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from fastapi import FastAPI, Request, Header, HTTPException
from fn import append_orc_msg, bay_ocr, bbl_ocr, convert_grayscale, get_img_size, get_ocr_locations, get_rois, gov_ocr, kbank_ocr, ktb_ocr, scb_ocr, tmb_ocr
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage


app = FastAPI()

line_id = os.getenv("LINE_BOT_BASIC_ID")
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
threshold = float(os.getenv("CLASSIFICATION_THRESHOLD"))

classification_model = tf.keras.models.load_model("./models/bank_classification_model.h5")
classification_labels = pickle.loads(open("./models/bank_classification_labels.pickle", "rb").read())

IMG_FILE_NAME = "./image.png"


@app.get("/")
def index():
    return {"line": line_id}


@app.post("/webhook")
async def callback(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="chatbot handle body error.")
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def message_text(event):
    print("text-event", event)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="กรุณาอัพโหลดรูป")
    )


@handler.add(MessageEvent, message=ImageMessage)
def message_text(event):
    print("image-event", event)

    # save image to file
    message_content = line_bot_api.get_message_content(event.message.id)
    with open(IMG_FILE_NAME, "wb") as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)

    # load image for classification
    img = Image.open(IMG_FILE_NAME).resize((180, 180))
    img_arr = np.array(img)

    # predict bank
    pred = classification_model.predict(img_arr[None, :, :])
    pred_prob = tf.nn.softmax(pred).numpy()

    # find bank class name and probability
    max_idx = np.argmax(pred_prob)
    max_prob = np.max(pred_prob)
    
    bank_class = classification_labels[max_idx] if max_prob > threshold else "OTHER"
    messages = []

    messages.append("bank: {}".format(bank_class))
    messages.append("prob: {}".format(max_prob))
    messages.append("-" * 10)

    if bank_class == "OTHER":
        messages.append("กรุณาอัพโหลดรูปสลิป")
    else:
        # load image for ocr
        img = cv2.imread(IMG_FILE_NAME)
        thr_img = convert_grayscale(img)

        if bank_class == "SCB":
            rois = get_rois(thr_img, 12, 0.1, 0.04)
            ocr = scb_ocr(rois)
            messages += append_orc_msg(ocr)

        elif bank_class == "GOV":
            rois = get_rois(thr_img, 10, 0.06, 0.04)
            ocr = gov_ocr(rois)
            messages += append_orc_msg(ocr)

        elif bank_class == "TMB":
            rois = get_rois(thr_img, 12, 0.1, 0.05)
            ocr = tmb_ocr(rois)
            messages += append_orc_msg(ocr)

        elif bank_class == "KTB":
            rois = get_rois(thr_img, 15, 0.2, 0.05)
            ocr = ktb_ocr(rois)
            messages += append_orc_msg(ocr)

        elif bank_class == "BBL":
            rois = get_rois(thr_img, 13, 0.2, 0.04)
            ocr = bbl_ocr(rois)
            messages += append_orc_msg(ocr)

        elif bank_class == "BAY":
            rois = get_rois(thr_img, 13, 0.2, 0.1)
            ocr = bay_ocr(rois)
            messages += append_orc_msg(ocr)

        elif bank_class == "KBANK":
            rois = get_rois(thr_img, 13, 0.2, 0.1)
            ocr = kbank_ocr(rois)
            messages += append_orc_msg(ocr)

        # else:
        #     # get locations for ocr
        #     ocr_locations = get_ocr_locations(bank_class)

        #     # ocr on each box
        #     try:
        #         for i in range(len(ocr_locations)):
        #             (x, y, w, h) = ocr_locations[i].bbox

        #             roi = img[y:y+h, x:x+w]

        #             lang = "eng" if i == len(ocr_locations) - 1 else "tha+eng"
        #             text = pytesseract.image_to_string(roi, lang=lang)

        #             messages.append("{}: {}".format(str(ocr_locations[i].id), text.strip()))
        #     except:
        #         print("something went wrong on ocr")

    combine_msgs = "\n".join(messages)
    msg = TextSendMessage(text=combine_msgs)
    line_bot_api.reply_message(event.reply_token, msg)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)