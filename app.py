#!/usr/bin/env python
# coding: utf-8

# In[32]:


import gradio as gr
import sqlite3
import numpy as np
from PIL import Image
import io
import uuid
import cv2


# In[34]:


# Initialize Database
DATABASE = 'Ashok_database.db'


# In[36]:


def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS images
                        (id TEXT PRIMARY KEY, image_blob BLOB)''')
    print("Database initialized.")

init_db()


# In[38]:


# Save image as .jpg in the database
def save_image_to_db(image_array, image_id):
    # Convert to .jpg format
    _, buffer = cv2.imencode('.jpg', image_array)
    
    # Save to database
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("INSERT INTO images (id, image_blob) VALUES (?, ?)", 
                     (image_id, buffer.tobytes()))
    print(f"Image saved with ID: {image_id}")


# In[40]:


# Convert Image to Numbers
def convert_image_to_numbers(image):
    image = Image.fromarray(image)  # Convert to PIL image
    image = image.resize((224, 224))  # Resize for consistency
    image_array = np.array(image)  # Convert to NumPy array
    np.save("image_data.npy", image_array)  # Save as .npy file
    return "Image converted and saved successfully!"


# In[42]:


# Retrieve image from the database
def retrieve_image_from_db(image_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.execute("SELECT image_blob FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        
    if row is None:
        return None
    
    # Convert binary data back to NumPy array
    image_array = np.frombuffer(row[0], dtype=np.uint8)
    
    # Decode image array to OpenCV format
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# In[44]:


# Convert PIL Image to OpenCV (NumPy array)
def pil_to_cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# In[46]:


# Convert OpenCV (NumPy array) to PIL Image
def cv2_to_pil(image_array):
    return Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))


# In[ ]:





# In[49]:


def upload(image):
    image_array = pil_to_cv2(image)
    image_id = str(uuid.uuid4())
    save_image_to_db(image_array, image_id)
    return f"Image successfully uploaded with ID: {image_id}"


# In[51]:


# Retrieve function to display image
def retrieve(image_id):
    image_array = retrieve_image_from_db(image_id)
    if image_array is None:
        return f"No image found with ID: {image_id}", None
    retrieved_image = cv2_to_pil(image_array)
    return f"Image successfully retrieved with ID: {image_id}", retrieved_image


# In[59]:


# Gradio Interfaces
upload_interface = gr.Interface(
    fn=upload,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Upload and Save Image as .jpg",
    description="Upload an image to save it as .jpg format in the database."
)

retrieve_interface = gr.Interface(
    fn=retrieve,
    inputs="text",
    outputs=["text", "image"],
    title="Retrieve Image from Database",
    description="Enter the unique ID to retrieve the saved image from the database."
)

# Combine Interfaces
app = gr.TabbedInterface([upload_interface, retrieve_interface], ["Upload", "Retrieve"])
app.launch(share=True)


# In[49]:





# In[ ]:




