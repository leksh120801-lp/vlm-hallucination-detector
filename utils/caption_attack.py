import random

OBJECT_SWAPS = {
    "dog": ["cat", "horse", "cow"],
    "cat": ["dog", "rabbit", "fox"],
    "car": ["bus", "truck", "motorcycle"],
    "person": ["robot", "statue", "dinosaur"]
}


def object_swap_attack(caption):

    words = caption.lower().split()

    for i, word in enumerate(words):

        if word in OBJECT_SWAPS:

            words[i] = random.choice(OBJECT_SWAPS[word])

            break

    return " ".join(words)

def generate_adversarial_captions(caption):

    attacks = []

    attacks.append(object_swap_attack(caption))

    attacks.append("a spaceship flying in space")

    attacks.append("a dinosaur walking in a city")

    return attacks