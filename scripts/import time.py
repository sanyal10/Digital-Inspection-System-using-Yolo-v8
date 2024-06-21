import time

def attention_test():
    print("Stay focused and press enter when you see the word 'RED'")
    time.sleep(2)  # wait for 2 seconds before starting
    start_time = time.time()
    while True:
        word = input("Enter the word: ")
        if word.lower() == "red":
            end_time = time.time()
            reaction_time = end_time - start_time
            print(f"Your reaction time: {reaction_time} seconds")
            break
        else:
            print("Try again!")

attention_test()
