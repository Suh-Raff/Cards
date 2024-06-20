from ultralytics import YOLO
import cv2
import cvzone
import math
import findScore

# I hard coded these classes, there are many ways to do this, I just decided to do it the fu** it way

classNames= ['10C', '10D', '10H', '10S',
             '2C', '2D', '2H', '2S',
             '3C', '3D', '3H', '3S',
             '4C', '4D', '4H', '4S',
             '5C', '5D', '5H', '5S',
             '6C', '6D', '6H', '6S',
             '7C', '7D', '7H', '7S',
             '8C', '8D', '8H', '8S',
             '9C', '9D', '9H', '9S',
             'AC', 'AD', 'AH', 'AS',
             'JC', 'JD', 'JH', 'JS',
             'KC', 'KD', 'KH', 'KS',
             'QC', 'QD', 'QH', 'QS']



# Webcam import  *****UNCOMENT THIS BLOCK TO USE WEB CAM MAKE SURE TO COMENT OUT THE VIDEO IMPORT BLOCK******
#cap = cv2.VideoCapture(2)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)

# Video import #Webcam import  *****UNCOMENT THIS BLOCK TO USE VIDEO FILE MAKE SURE TO COMENT OUT THE WEBCAM IMPORT BLOCK******
# I uploaded some test videos on GitHub, but it should work on any mp4
video_file = cv2.VideoCapture('../videos/player_v_dealer720.mp4')

# Load the YOLO trained model from Yolo-Weights fite (Check Github)
model = YOLO("../Models/playingCards.pt")

# Create masks
dealer_mask = cv2.imread('../images/dealer_mask720.png')
player_mask = cv2.imread('../images/player_mask720.png')
deck = []  # Empty list to keep track of all the cards dealt (player and dealer)

# Loop to display images, add boxes and labels
while True:
    success, img = video_file.read()  # Read image from video file
    imgDealerRegion = cv2.bitwise_and(img, dealer_mask)  # Do a bitwise addition of dealer_mask and image
    imgPlayerRegion = cv2.bitwise_and(img, player_mask)  # Do a bitwise operation of player mask and image
    results_dealer = model(imgDealerRegion, stream=True)  # Feed masked image so it only detects dealers hand
    results_player = model(imgPlayerRegion, stream=True)  # Feed masked image so it only detects players hand
    player_hand = []  # Empty list for players hand
    dealer_hand = []  # Empty list for dealers hand


    # Run detection for dealer
    for r in results_dealer:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # print(x1, y1, x2, y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))

            # Confidence value display
            conf = math.ceil((box.conf[0] * 100)) / 100
            #print(conf)

            # Class name (From hard coded classes, Doesn't work too good, but it works)
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.8)

            if conf > 0.5:
                dealer_hand.append(classNames[cls])  # Add the detected card to the dealer_hand list
                deck.append(classNames[cls])  # Add the card to the deck list

    # Run detection for player
    for r in results_player:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # print(x1, y1, x2, y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))

            # Confidence value display
            conf = math.ceil((box.conf[0] * 100)) / 100
            #print(conf)

            # Class name (From hard coded classes, Doesn't work too good, but it works)
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1.8)

            if conf > 0.5:
                player_hand.append(classNames[cls])  # Add card to player_hand list
                deck.append(classNames[cls])  # Add card to deck list

    # Turn lists into set to delete duplicates
    player_hand = list(set(player_hand))
    # print(player_hand) # Print statement for testing
    dealer_hand = list(set(dealer_hand))
    # print(dealer_hand)  # Print statement for testing
    deck = list(set(deck))
    # print(f'deck: {deck}')  # Print statement for testing

    # Get player score and print results to screen
    player_results = findScore.findBlackjackScore(player_hand)
    print(player_results)
    if player_results <= 21:
        cvzone.putTextRect(img, f'Player: {player_results} ', (20, 50), scale=1.8)
    else:
        cvzone.putTextRect(img, 'Player Bust', (20, 50), scale=1.8)

    # Get dealer score and print results to screen
    dealer_results = findScore.findBlackjackScore(dealer_hand)
    print(dealer_results)
    if dealer_results <= 21:
        cvzone.putTextRect(img, f'Dealer: {dealer_results} ', (20, 80), scale=1.8)
    else:
        cvzone.putTextRect(img, 'Dealer Bust', (20, 80), scale=1.8)

    # Get the total deck count and show on screen
    count = findScore.running_count(deck)
    # print(f'count: {count}') # Print statement for testing
    cvzone.putTextRect(img, f'Running Count: {count}', (20, 110), scale=1.8)

    # Display the video
    cv2.imshow('Image', img)
    cv2.waitKey(1)