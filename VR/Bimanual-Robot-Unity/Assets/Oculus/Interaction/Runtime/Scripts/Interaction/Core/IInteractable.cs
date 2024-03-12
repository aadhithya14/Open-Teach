/************************************************************************************
Copyright : Copyright (c) Facebook Technologies, LLC and its affiliates. All rights reserved.

Your use of this SDK or tool is subject to the Oculus SDK License Agreement, available at
https://developer.oculus.com/licenses/oculussdk/

Unless required by applicable law or agreed to in writing, the Utilities SDK distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.
************************************************************************************/

using System;

namespace Oculus.Interaction
{
    public struct InteractableStateChangeArgs
    {
        public InteractableState PreviousState;
        public InteractableState NewState;
    }

    /// <summary>
    /// An IInteractableView defines the view for an object that can be
    /// interacted with.
    /// </summary>
    public interface IInteractableView
    {
        InteractableState State { get; }
        event Action<InteractableStateChangeArgs> WhenStateChanged;

        int MaxInteractors { get; }
        int MaxSelectingInteractors { get; }

        int InteractorsCount { get; }
        int SelectingInteractorsCount { get; }

        event Action WhenInteractorsCountUpdated;
        event Action WhenSelectingInteractorsCountUpdated;
    }

    /// <summary>
    /// An object that can be interacted with, an IInteractable can, in addition to
    /// an IInteractableView, be enabled or disabled.
    /// </summary>
    public interface IInteractable : IInteractableView
    {
        void Enable();
        void Disable();
        new int MaxInteractors { get; set; }
        new int MaxSelectingInteractors { get; set; }
    }
}
