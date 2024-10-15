import tqdm
import time  # For demonstration of elapsed time

# Example of how your training loop might look
for epoch in range(epochs):
    correct = 0.0
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("\n\nTraining...")
    
    mvnetwork.train()
    mvtn.train()
    mvrenderer.train()
    
    running_loss = 0
    # Use tqdm to create a progress bar
    with tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
        for i, (targets, meshes, points) in enumerate(train_loader):
            start_time = time.time()  # Start timing the batch processing

            azim, elev, dist = mvtn(points, c_batch_size=len(targets))
            rendered_images, _ = mvrenderer(meshes, points, azim=azim, elev=elev, dist=dist)
            outputs = mvnetwork(rendered_images)[0]

            loss = criterion(outputs, targets.cuda())
            running_loss += loss.item()
            loss.backward()
            correct += (torch.max(outputs, dim=1)[1] == targets.cuda()).to(torch.int32).sum().item()
            optimizer.step()
            optimizer.zero_grad()

            if mvtn_optimizer is not None:
                mvtn_optimizer.step()
                mvtn_optimizer.zero_grad()

            # Update tqdm progress bar
            elapsed_time = time.time() - start_time
            pbar.set_postfix({'loss': running_loss / (i + 1), 'accuracy': 100.0 * correct / ((i + 1) * len(targets))})
            pbar.update(1)  # Increment progress bar
            
            # Optionally, update the description to include elapsed time or ETA
            pbar.desc = f'Epoch {epoch + 1}/{epochs} | Elapsed: {elapsed_time:.2f}s'

    print(f"\nAverage Training Loss = {(running_loss / len(train_loader)):.5f}. Average Training Accuracy = {(100.0 * correct / len(dset_train)):.2f}.")

