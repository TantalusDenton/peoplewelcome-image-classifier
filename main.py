from typing import List
from fastapi import FastAPI, HTTPException
from models import User, Gender, Role, UserUpdateRequest, Image
from uuid import UUID, uuid4
import IsThisAPigeonClassifier

#Start FastAPI
app = FastAPI()

db: List[User] = [
    User(
        id = UUID("4981100f-f377-440f-8985-64fa0f0ff5a2"),
        first_name = "Joe",
        last_name = "Zimmerman",
        gender = Gender.male,
        roles = [Role.admin, Role.user]
    ),

    User(
        id = uuid4(),
        first_name = "Agent",
        last_name = "Smith",
        gender = Gender.male,
        roles = [Role.user]
    )
]

images: List[Image] = [
    Image(
        image_id = 0,
        label = IsThisAPigeonClassifier.labels[0]
        #score = IsThisAPigeonClassifier.scores[0]
    ),
    Image(
        image_id = 1,
        label = IsThisAPigeonClassifier.labels[1]
        #score = IsThisAPigeonClassifier.scores[1]
    )
    ,
    Image(
        image_id = 2,
        label = IsThisAPigeonClassifier.labels[2]
        #score = IsThisAPigeonClassifier.scores[2]
    ),
    Image(
        image_id = 3,
        label = IsThisAPigeonClassifier.labels[3]
        #score = IsThisAPigeonClassifier.scores[3]
    ),
    Image(
        image_id = 4,
        label = IsThisAPigeonClassifier.labels[4]
        #score = IsThisAPigeonClassifier.scores[4]
    ),
    Image(
        image_id = 5,
        label = IsThisAPigeonClassifier.labels[5],
        score = IsThisAPigeonClassifier.scores[5]
    ),
    Image(
        image_id = 6,
        label = IsThisAPigeonClassifier.labels[6]
        #score = IsThisAPigeonClassifier.scores[6]
    ),
    Image(
        image_id = 7,
        label = IsThisAPigeonClassifier.labels[7]
        #score = IsThisAPigeonClassifier.scores[7]
    ),
    Image(
        image_id = 8,
        label = IsThisAPigeonClassifier.labels[8]
        #score = IsThisAPigeonClassifier.scores[8]
    ),
    Image(
        image_id = 9,
        label = IsThisAPigeonClassifier.labels[9]
        #score = IsThisAPigeonClassifier.scores[9]
    )
]

@app.get("/")
async def root():
    #await foo()
    return {"Hello": "Mundo"}

@app.get("/api/v1/users")
async def fetch_user():
    return db

@app.get("/api/v1/image-classifier")
async def fetch_label():
    return images

@app.get("/api/v1/image/{image_id}")
async def fetch_label():
    return images

@app.post("/api/v1/image/")
async def infer_label(image: Image):
    image.label = IsThisAPigeonClassifier.infer(image.image_id)
    images.append(image)
    return {"label": image.label}

@app.post("/api/v1/users")
async def register_user(user: User):
    db.append(user)
    return {"id": user.id}

@app.delete("/api/v1/users/{user_id}")
async def delete_user(user_id: UUID):
    for user in db:
        if user.id == user_id:
            db.remove(user)
            return
    raise HTTPException(
        status_code = 404,
        detail = f"user with id: {user_id} does not exist"
    )

@app.put("/api/v1/users/{user_id}")
async def update_user(user_update: UserUpdateRequest, user_id: UUID):
    for user in db:
        if user.id == user_id:
            if user_update.first_name is not None:
                user.first_name = user_update.first_name
            return   
    raise HTTPException(
        status_code = 404,
        detail = f"user with id: {user_id} does not exist"
    )