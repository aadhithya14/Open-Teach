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
    public struct InteractorStateChangeArgs
    {
        public InteractorState PreviousState;
        public InteractorState NewState;
    }

    /// <summary>
    /// IInteractorView defines the view for an object that can interact with other objects.
    /// </summary>
    public interface IInteractorView
    {
        int Identifier { get; }

        bool HasCandidate { get; }
        object Candidate { get; }

        bool HasInteractable { get; }
        bool HasSelectedInteractable { get; }

        InteractorState State { get; }
        event Action<InteractorStateChangeArgs> WhenStateChanged;
        event Action WhenPreprocessed;
        event Action WhenProcessed;
        event Action WhenPostprocessed;
    }

    /// <summary>
    /// IInteractor defines an object that can interact with other objects
    /// and can handle selection events to change its state.
    /// </summary>
    public interface IInteractor : IInteractorView
    {

        void Preprocess();
        void Process();
        void Postprocess();

        void ProcessCandidate();
        void Enable();
        void Disable();
        void Hover();
        void Unhover();
        void Select();
        void Unselect();

        bool ShouldHover { get; }
        bool ShouldUnhover { get; }
        bool ShouldSelect { get; }
        bool ShouldUnselect { get; }

        bool IsRootDriver { get; set; }
    }
}
