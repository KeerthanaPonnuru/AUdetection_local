bbox,landmarks = detect_faces(img)
if len(bbox)==0:
return None
bbox = bbox[0,:-1]
p = bbox[2:] - bbox[:2]
bbox[:2]-=p;bbox[2:]+=p
cropped = img.crop(bbox)